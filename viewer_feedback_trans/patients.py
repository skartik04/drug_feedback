REVIEW_PATIENTS = [
    # poly→mono correct (4)
    {"pid": "109_Nakazi Hope",              "category": "poly_mono_correct"},
    {"pid": "17_Kabito Alpha",              "category": "poly_mono_correct"},
    {"pid": "71_Nabbossa Meghan",           "category": "poly_mono_correct"},
    {"pid": "72_Nalusiba Sandra",           "category": "poly_mono_correct"},
    # mono→poly correct (1)
    {"pid": "107_Ismahan Mahad",            "category": "mono_poly_correct"},
    # mono→poly V1→V2 wrong (8)
    {"pid": "167_Asiimwe Roland",           "category": "mono_poly_v12_wrong"},
    {"pid": "119_Odur Hillary",             "category": "mono_poly_v12_wrong"},
    {"pid": "307_Sembatya Tian",            "category": "mono_poly_v12_wrong"},
    {"pid": "16_Katoosi Timothy",           "category": "mono_poly_v12_wrong"},
    {"pid": "86_Namukose Gift",             "category": "mono_poly_v12_wrong"},
    {"pid": "51_Nakafeero Pauline",         "category": "mono_poly_v12_wrong"},
    {"pid": "158_Namugalu Shatra",          "category": "mono_poly_v12_wrong"},
    {"pid": "155_Numazi Shamirah",          "category": "mono_poly_v12_wrong"},
    # mono→poly V2→V3 wrong (7)
    {"pid": "53_Bukenya Patrick",           "category": "mono_poly_v23_wrong"},
    {"pid": "182_Lubega Joel",              "category": "mono_poly_v23_wrong"},
    {"pid": "4_Asio Esther Jane",           "category": "mono_poly_v23_wrong"},
    {"pid": "7_Akello Angel",               "category": "mono_poly_v23_wrong"},
    {"pid": "327_Ssekamatta Darlene Rose",  "category": "mono_poly_v23_wrong"},
    {"pid": "38_Nansera George",            "category": "mono_poly_v23_wrong"},
    {"pid": "199_Mayamba Jedidah",          "category": "mono_poly_v23_wrong"},
]

CATEGORY_LABELS = {
    "poly_mono_correct":   "Poly→Mono · Correct",
    "mono_poly_correct":   "Mono→Poly · Correct",
    "mono_poly_v12_wrong": "Mono→Poly V1→V2 · Wrong",
    "mono_poly_v23_wrong": "Mono→Poly V2→V3 · Wrong",
}
