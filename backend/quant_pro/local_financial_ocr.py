"""Local OCR extractor for NEPSE financial statement filings.

This pipeline is designed for table-heavy annual and quarterly statements with
mixed English/Nepali labels and watermark noise. It prioritizes deterministic
extraction of the trading-agent-relevant fields and always emits validation
flags instead of pretending low-confidence OCR is trustworthy.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - import guard for environments without cv2
    cv2 = None


DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

CORE_FIELDS = [
    "income_statement.total_revenue",
    "income_statement.net_profit",
    "per_share.eps",
    "per_share.book_value",
]

DEFAULT_LINE_MATCH_THRESHOLD = 0.72
DEFAULT_ROW_MATCH_THRESHOLD = 0.52
MAX_OCR_DIMENSION = 1600

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "income_statement.total_revenue": (
        "revenue from sale of electricity",
        "total revenue",
        "revenue",
        "total operating income",
        "operating income",
        "income from sales",
        "interest income",
        "net interest income",
        "बिक्रीबाट आम्दानी",
        "राजश्व",
        "कुल आम्दानी",
        "कुल सञ्चालन आम्दानी",
        "खुद ब्याज शुल्क तथा कमिशन आम्दानी",
    ),
    "income_statement.operating_profit": (
        "profit from operation",
        "profit from operation",
        "operating profit",
        "operating profit/loss",
        "सञ्चालन मुनाफा",
        "सञ्चालन नाफा",
    ),
    "income_statement.net_profit": (
        "net profit",
        "profit for the period",
        "profit after tax",
        "net profit/loss",
        "profit loss for the period",
        "कर पछिको नाफा",
        "शुद्ध नाफा",
        "खुद नाफा",
        "नाफा",
    ),
    "balance_sheet.total_assets": (
        "total assets",
        "कुल सम्पत्ति",
        "जम्मा सम्पत्ति",
    ),
    "balance_sheet.total_liabilities": (
        "total liabilities",
        "कुल दायित्व",
        "जम्मा दायित्व",
    ),
    "balance_sheet.shareholders_equity": (
        "total equity",
        "shareholders equity",
        "equity",
        "कुल इक्विटी",
        "बैंकका शेयरधनीहरुको कुल पूँजी",
        "शेयरधनीकोष",
    ),
    "balance_sheet.share_capital": (
        "share capital",
        "paid up capital",
        "चुक्ता पूँजी",
        "शेयर पूँजी",
    ),
    "balance_sheet.retained_earnings": (
        "retained earnings",
        "retained profit",
        "retained earning",
        "सञ्चित मुनाफा",
    ),
    "balance_sheet.total_deposits": (
        "deposits from customers",
        "total deposits",
        "deposits",
        "निक्षेप",
    ),
    "balance_sheet.total_loans": (
        "loans and advances to customers",
        "loans and advances",
        "total loan",
        "कर्जा तथा सापटी",
        "कर्जा",
    ),
    "per_share.eps": (
        "basic earnings per share",
        "earnings per share",
        "eps",
        "प्रति शेयर आम्दानी",
    ),
    "per_share.book_value": (
        "book value",
        "net worth per share",
        "book value per share",
        "प्रति शेयर नेटवर्थ",
        "प्रति शेयर सम्पत्तिको मूल्य",
    ),
    "ratios.npl_pct": (
        "npl",
        "non performing loan",
        "नन परफर्मिङ लोन",
    ),
    "ratios.capital_adequacy_pct": (
        "capital adequacy",
        "car",
        "capital to risk weighted assets ratio",
        "पूँजी कोष पर्याप्तता",
    ),
}


def _normalize_text(text: str) -> str:
    text = text.translate(DEVANAGARI_DIGITS).lower()
    text = re.sub(r"[_|]+", " ", text)
    text = re.sub(r"[^0-9a-zऀ-ॿ%./()\-+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_number_token(token: str) -> float | None:
    token = token.translate(DEVANAGARI_DIGITS)
    token = token.replace("O", "0").replace("o", "0")
    token = token.replace("I", "1").replace("l", "1")
    token = token.replace("B", "8")
    token = token.replace("S", "5")
    token = token.replace(" ", "")
    if not token:
        return None
    neg = token.startswith("(") and token.endswith(")")
    token = token.strip("()")
    token = token.replace(",", "")
    if token.count(".") > 1:
        first, *rest = token.split(".")
        token = first + "." + "".join(rest)
    try:
        value = float(token)
    except ValueError:
        return None
    if neg:
        value = -value
    return value


def _extract_numbers(text: str) -> list[float]:
    normalized = text.translate(DEVANAGARI_DIGITS)
    pattern = r"\(?[0-9OoIlSB][0-9OoIlSB,.\-()]*\)?"
    values: list[float] = []
    for match in re.finditer(pattern, normalized):
        token = match.group(0)
        if sum(ch.isdigit() for ch in token.translate(DEVANAGARI_DIGITS)) < 2:
            continue
        value = _normalize_number_token(token)
        if value is not None:
            values.append(value)
    return values


def _select_field_value(field: str, numbers: list[float]) -> float | None:
    if not numbers:
        return None
    if field.startswith("ratios."):
        for value in numbers:
            if 0 <= abs(value) <= 100:
                return value
        return numbers[0]
    if field.startswith("per_share."):
        for value in numbers:
            if abs(value) <= 100_000:
                return value
        return numbers[0]
    for value in numbers:
        if abs(value) >= 1:
            return value
    return numbers[0]


def _render_pages(path: Path, max_pages: int = 2) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(path)
        pages = []
        for idx in range(min(max_pages, len(doc))):
            page = doc[idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        return pages
    image = Image.open(path)
    frames = []
    for idx in range(min(max_pages, getattr(image, "n_frames", 1))):
        image.seek(idx)
        frames.append(image.convert("RGB"))
    return frames


def _downscale_for_ocr(image: Image.Image, max_dimension: int = MAX_OCR_DIMENSION) -> Image.Image:
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_dimension:
        return image
    scale = max_dimension / current_max
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for local OCR extraction")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _prepare_gray(image: Image.Image) -> np.ndarray:
    cv_img = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def _binary_variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    variants: dict[str, np.ndarray] = {}
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    inverted = 255 - otsu
    variants["gray"] = gray
    variants["otsu"] = otsu
    variants["adaptive"] = adaptive
    variants["inverted"] = inverted
    return variants


def _ocr_image(
    image: np.ndarray | Image.Image,
    lang: str,
    config: str,
) -> str:
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    return pytesseract.image_to_string(pil, lang=lang, config=config)


def _split_ocr_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if len(cleaned) < 4:
            continue
        lines.append(cleaned)
    return lines


def _detect_row_bands(binary: np.ndarray) -> list[tuple[int, int]]:
    img = 255 - binary if binary.mean() > 127 else binary
    kernel_width = max(30, img.shape[1] // 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = lines.sum(axis=1)
    threshold = projection.max() * 0.15 if projection.max() > 0 else 0
    ys = [idx for idx, value in enumerate(projection) if value > threshold]
    if not ys:
        return []
    merged = []
    start = ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y - prev <= 3:
            prev = y
            continue
        merged.append((start, prev))
        start = prev = y
    merged.append((start, prev))

    bands: list[tuple[int, int]] = []
    for (_, upper), (lower, _) in zip(merged, merged[1:]):
        y1 = max(0, upper - 2)
        y2 = min(binary.shape[0], lower + 2)
        if 20 <= y2 - y1 <= 180:
            bands.append((y1, y2))
    return bands


def _best_field_for_label(label_norm: str, target_fields: set[str] | None = None) -> tuple[str | None, float]:
    best_field = None
    best_score = 0.0
    fields = target_fields or set(FIELD_ALIASES)
    for field in fields:
        for alias in FIELD_ALIASES[field]:
            alias_norm = _normalize_text(alias)
            if alias_norm in label_norm or label_norm in alias_norm:
                score = 1.0
            else:
                score = SequenceMatcher(None, label_norm, alias_norm).ratio()
            if score > best_score:
                best_field = field
                best_score = score
    return best_field, best_score


def _row_candidates(image: Image.Image, target_fields: set[str] | None = None) -> list[dict[str, Any]]:
    gray = _prepare_gray(image)
    variants = _binary_variants(gray)
    binary = variants["adaptive"]
    bands = _detect_row_bands(binary)
    if not bands:
        return []

    width = binary.shape[1]
    left_cut = int(width * 0.52)
    rows: list[dict[str, Any]] = []
    for y1, y2 in bands:
        row_img = binary[y1:y2, :]
        label_img = row_img[:, :left_cut]
        label_text = _ocr_image(label_img, lang="eng+nep", config="--psm 7")
        if not label_text.strip():
            label_text = _ocr_image(label_img, lang="eng+nep", config="--psm 6")
        label_norm = _normalize_text(label_text)
        if target_fields:
            _, score = _best_field_for_label(label_norm, target_fields)
            if score < 0.46:
                continue
        value_img = row_img[:, left_cut:]
        value_text = _ocr_image(
            value_img,
            lang="eng",
            config="--psm 7 -c tessedit_char_whitelist=0123456789.,()-",
        )
        numbers = _extract_numbers(value_text)
        rows.append(
            {
                "y1": y1,
                "y2": y2,
                "label_raw": label_text.strip(),
                "label_norm": label_norm,
                "values_raw": value_text.strip(),
                "numbers": numbers,
            }
        )
    return rows


def _best_match(
    candidates: list[dict[str, Any]],
    aliases: tuple[str, ...],
    *,
    text_key: str,
) -> tuple[dict[str, Any] | None, float]:
    best_row = None
    best_score = 0.0
    for row in candidates:
        label = row[text_key]
        if not label:
            continue
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm in label or label in alias_norm:
                score = 1.0
            else:
                score = SequenceMatcher(None, label, alias_norm).ratio()
            if score > best_score:
                best_row = row
                best_score = score
    return best_row, best_score


def _infer_sector(text: str, values: dict[str, Any]) -> str:
    normalized = _normalize_text(text)
    if values["balance_sheet"].get("total_deposits") or values["balance_sheet"].get("total_loans"):
        return "banking"
    if "sale of electricity" in normalized or "विद्युत" in normalized:
        return "hydropower"
    if "insurance" in normalized or "बीमा" in normalized:
        return "insurance"
    return "other"


def _set_nested(container: dict[str, Any], dotted: str, value: Any) -> None:
    left, right = dotted.split(".", 1)
    container[left][right] = value


def _empty_values() -> dict[str, Any]:
    return {
        "balance_sheet": {
            "total_assets": 0,
            "total_liabilities": 0,
            "shareholders_equity": 0,
            "share_capital": 0,
            "retained_earnings": 0,
            "total_deposits": 0,
            "total_loans": 0,
        },
        "income_statement": {
            "total_revenue": 0,
            "operating_profit": 0,
            "net_profit": 0,
            "interest_income": 0,
            "interest_expense": 0,
        },
        "per_share": {
            "eps": 0,
            "book_value": 0,
        },
        "ratios": {
            "npl_pct": 0,
            "capital_adequacy_pct": 0,
            "cost_income_ratio": 0,
        },
    }


def _build_line_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in _split_ocr_lines(text):
        candidates.append(
            {
                "raw": line,
                "label_norm": _normalize_text(line),
                "numbers": _extract_numbers(line),
            }
        )
    return candidates


def _extract_candidates(
    candidates: list[dict[str, Any]],
    *,
    threshold: float,
    source: str,
) -> tuple[dict[str, Any], dict[str, float], list[str], list[str]]:
    values = _empty_values()
    field_scores: dict[str, float] = {}
    matched_fields: list[str] = []
    flags: list[str] = []

    for field, aliases in FIELD_ALIASES.items():
        candidate, score = _best_match(candidates, aliases, text_key="label_norm")
        if not candidate or score < threshold:
            continue
        value = _select_field_value(field, candidate["numbers"])
        if value is None:
            flags.append(f"{source}:{field}:matched_label_but_no_numeric_value")
            continue
        if field.startswith("ratios.") and value > 1_000:
            value = value / 100
        _set_nested(values, field, value)
        field_scores[field] = score
        matched_fields.append(field)

    return values, field_scores, matched_fields, flags


def _extract_from_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float], list[str], list[str]]:
    return _extract_candidates(rows, threshold=DEFAULT_ROW_MATCH_THRESHOLD, source="row")


def _extract_from_lines(text: str) -> tuple[dict[str, Any], dict[str, float], list[str], list[str]]:
    return _extract_candidates(
        _build_line_candidates(text),
        threshold=DEFAULT_LINE_MATCH_THRESHOLD,
        source="line",
    )


def _flatten_values(values: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for outer_key, inner in values.items():
        for inner_key, value in inner.items():
            flat[f"{outer_key}.{inner_key}"] = value
    return flat


def _merge_extractions(
    line_values: dict[str, Any],
    line_scores: dict[str, float],
    row_values: dict[str, Any],
    row_scores: dict[str, float],
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    merged = _empty_values()
    review_flags: list[str] = []
    chosen_source: dict[str, str] = {}
    line_flat = _flatten_values(line_values)
    row_flat = _flatten_values(row_values)

    for field in FIELD_ALIASES:
        line_value = line_flat.get(field, 0)
        row_value = row_flat.get(field, 0)
        line_score = line_scores.get(field, 0.0)
        row_score = row_scores.get(field, 0.0)

        selected_value = 0
        selected_source = ""

        if row_value and (not line_value or row_score >= line_score * 0.9):
            selected_value = row_value
            selected_source = "row"
        elif line_value:
            selected_value = line_value
            selected_source = "line"
        elif row_value:
            selected_value = row_value
            selected_source = "row"

        if line_value and row_value:
            base = max(abs(line_value), abs(row_value), 1)
            if abs(line_value - row_value) / base > 0.18:
                review_flags.append(f"inconsistent_value:{field}")

        if selected_value:
            _set_nested(merged, field, selected_value)
            chosen_source[field] = selected_source

    return merged, review_flags, chosen_source


def _collect_full_text(images: list[Image.Image]) -> str:
    parts = []
    for image in images:
        gray = ImageOps.autocontrast(image.convert("L"))
        width, height = gray.size
        crops = [
            gray,
            gray.crop((0, int(height * 0.45), width, height)),
        ]
        for crop in crops:
            parts.append(_ocr_image(crop, lang="eng+nep", config="--psm 6"))
    return "\n".join(parts)


def _validate(values: dict[str, Any], matched_fields: list[str], flags: list[str]) -> tuple[float, list[str]]:
    review = list(flags)
    bs = values["balance_sheet"]
    inc = values["income_statement"]
    ps = values["per_share"]

    assets = bs.get("total_assets") or 0
    liabilities = bs.get("total_liabilities") or 0
    equity = bs.get("shareholders_equity") or 0
    if assets and liabilities and equity:
        gap = abs((liabilities + equity) - assets) / max(assets, 1)
        if gap > 0.35:
            review.append(f"balance_sheet_mismatch:{gap:.2f}")

    revenue = inc.get("total_revenue") or 0
    net_profit = inc.get("net_profit") or 0
    if revenue and abs(net_profit) > revenue * 4:
        review.append("net_profit_unusually_large_vs_revenue")

    if ps.get("eps") and abs(ps["eps"]) > 5000:
        review.append("eps_implausible")
    if ps.get("book_value") and abs(ps["book_value"]) > 100000:
        review.append("book_value_implausible")

    for field in CORE_FIELDS:
        if field not in matched_fields:
            review.append(f"missing_core_field:{field}")

    score = len(matched_fields) / max(len(CORE_FIELDS) + 6, 1)
    score = min(1.0, 0.35 + score)
    if any(flag.startswith("missing_core_field") for flag in review):
        score -= 0.2
    if any("mismatch" in flag or "implausible" in flag for flag in review):
        score -= 0.15
    return max(0.0, min(1.0, score)), review


def extract_financials_locally(path: Path | str, description: str = "") -> dict[str, Any]:
    """Extract financial fields locally using OpenCV + Tesseract."""
    path = Path(path)
    images = [_downscale_for_ocr(image) for image in _render_pages(path, max_pages=2)]
    full_text = _collect_full_text(images)
    line_values, line_scores, line_matches, line_flags = _extract_from_lines(full_text)
    rows: list[dict[str, Any]] = []
    row_values = _empty_values()
    row_scores: dict[str, float] = {}
    row_matches: list[str] = []
    row_flags: list[str] = []

    missing_core_fields = {field for field in CORE_FIELDS if field not in line_matches}
    if missing_core_fields:
        for image in images:
            rows.extend(_row_candidates(image, target_fields=missing_core_fields))
        row_values, row_scores, row_matches, row_flags = _extract_from_rows(rows)

    values, merge_flags, chosen_source = _merge_extractions(
        line_values,
        line_scores,
        row_values,
        row_scores,
    )
    matched_fields = sorted({*line_matches, *row_matches})
    flags = [*line_flags, *row_flags, *merge_flags]
    period = {"fiscal_year": None, "quarter": None}
    try:
        from backend.quant_pro.data_scrapers.financial_reports import parse_report_period

        period = parse_report_period(description) or period
        if not period.get("fiscal_year") or not period.get("quarter"):
            ocr_period = parse_report_period(full_text)
            period = {
                "fiscal_year": period.get("fiscal_year") or ocr_period.get("fiscal_year"),
                "quarter": period.get("quarter") or ocr_period.get("quarter"),
            }
    except Exception:
        pass

    sector = _infer_sector(full_text + "\n" + description, values)
    confidence, review_flags = _validate(values, matched_fields, flags)

    notes = []
    if description:
        notes.append(description[:180])
    if matched_fields:
        notes.append("matched: " + ", ".join(sorted(matched_fields)[:8]))
    if chosen_source:
        notes.append(
            "source: "
            + ", ".join(f"{field}={source}" for field, source in sorted(chosen_source.items())[:8])
        )

    result = {
        "sector": sector,
        "fiscal_year": period.get("fiscal_year") or "",
        "quarter": f"Q{period['quarter']}" if period.get("quarter") else "",
        "balance_sheet": values["balance_sheet"],
        "income_statement": values["income_statement"],
        "per_share": values["per_share"],
        "ratios": values["ratios"],
        "notes": " | ".join(notes),
        "quality": {
            "engine": "local_tesseract_opencv",
            "confidence": round(confidence, 3),
            "matched_fields": sorted(matched_fields),
            "review_flags": review_flags,
            "line_match_count": len(line_matches),
            "row_match_count": len(row_matches),
            "used_row_ocr": bool(rows),
        },
        "ocr_text_preview": full_text[:4000],
    }
    return result
