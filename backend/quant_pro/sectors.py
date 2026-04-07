"""
Sector group mappings used for building benchmark series.
"""

SECTOR_GROUPS = {
    "Commercial Banks": {
        "ADBL", "CZBIL", "EBL", "GBIME", "HBL", "KBL", "LSL", "MBL",
        "NABIL", "NBL", "NICA", "NIMB", "NMB", "PCBL", "PRVU", "RBB",
        "SANIMA", "SBI", "SBL", "SCB",
    },
    "Development Banks": {
        "CORBL", "EDBL", "GBBL", "GRDBL", "JBBL", "KRBL", "KSBBL", "LBBL",
        "MDB", "MLBL", "MNBBL", "NABBC", "SABBL", "SADBL", "SAPDBL", "SHINE",
        "SINDU",
    },
    "Finance": {
        "BFC", "CFCL", "GFCL", "GMFIL", "GUFL", "ICFC", "JFL", "MFIL",
        "MPFL", "NFS", "PFL", "PROFL", "RLFL", "SFCL", "SIFC",
    },
    "Hydropower": {
        "AHL", "AHPC", "AKJCL", "AKPL", "API", "BARUN", "BEDC", "BGWT",
        "BHCL", "BHDC", "BHL", "BHPL", "BJHL", "BNHC", "BPCL", "BUNGAL",
        "CHCL", "CHDC", "CHL", "CKHL", "DHEL", "DHPL", "DOLTI", "DORDI",
        "EHPL", "GHL", "GLH", "GVL", "HDHPC", "HHL", "HIMSTAR", "HPPL",
        "HURJA", "IHL", "JHAPA", "JOSHI", "KKHC", "KPCL", "LEC", "MABEL", "MAKAR",
        "MANDU", "MBJC", "MCHL", "MEHL", "MEL", "MEN", "MHCL", "MHL",
        "MHNL", "MKHL", "MKHC", "MKJC", "MMKJL", "MSHL", "NGPL", "NHDL",
        "NHPC", "NYADI", "PHCL", "PMHPL", "PPCL", "PPL", "PURE", "RADHI",
        "RAWA", "RFPL", "RHGCL", "RHPL", "RIDI", "RURU", "SAHAS", "SANVI",
        "SGHC", "SHEL", "SHPC", "SIKLES", "SJCL", "SKHL", "SMH", "SMHL",
        "SMJC", "SOHL", "SPC", "SPDL", "SPHL", "SPL", "SSHL", "TAMOR",
        "TPC", "TSHL", "TVCL", "UHEWA", "ULHC", "UMHL", "UMRH", "UNHPL",
        "UPCL", "UPPER", "USHEC", "USHL", "VLUCL",
    },
    "Life Insurance": {
        "ALICL", "CLI", "CREST", "GMLI", "HLI", "ILI", "IMEL", "LICN",
        "NATRIS", "NLIC", "NLICL", "PMLI", "RNLI", "SJLIC", "SNLI", "SRLI",
    },
    "Non-Life Insurance": {
        "HEI", "HGI", "IGI", "IGIPR", "NECO", "NICL", "NIL", "NLG",
        "NMIC", "PRIN", "RBCL", "SALICO", "SGIC", "SIC", "SICL", "SPIL",
        "UAIL",
    },
    "Microfinance": {
        "ACLBSL", "ALBSL", "AVYAN", "CBBL", "CYCL", "DDBL", "DLBS", "FMDBL",
        "FOWAD", "GBLBS", "GILB", "GLBSL", "GMFBS", "HLBSL", "ILBS", "JBLB",
        "JSLBB", "KMCDB", "LLBS", "MATRI", "MERO", "MLBBL", "MLBS", "MLBSL",
        "MSLB", "NADEP", "NICLBSL", "NMBMF", "NMFBS", "NMLBBL", "NUBL",
        "NESDO", "RSDC", "SHLB", "SKBBL", "SLBBL", "SLBSL", "SMB", "SMATA",
        "SMFBS", "SMFDB", "SMPDA", "SWASTIK", "SWBBL", "SWMF", "ULBSL",
        "UNLB", "USLB", "VLBS", "WNLB",
    },
    "Hotels & Tourism": {"BANDIPUR", "CGH", "CITY", "KDL", "OHL", "SHL", "TRH", "TTL"},
    "Manufacturing & Processing": {
        "BNT", "ENL", "GCIL", "HDL", "MKCL", "NLO", "OMPL", "RSML",
        "SAGAR", "SAIL", "SARBTM", "SHIVM", "SONA", "SYPNL", "UNL",
    },
    "Investment": {"CIT", "HATHY", "HIDCL", "NIFRA", "NRM", "NRN", "NWCL"},
    "Others": {"HRL", "MKHL", "NRIC", "NTC"},
    "Trading": {"BBC", "STC"},
}

SECTOR_LOOKUP = {
    symbol: sector for sector, members in SECTOR_GROUPS.items() for symbol in members
}

__all__ = ["SECTOR_GROUPS", "SECTOR_LOOKUP"]
