"""Medical specialties and available doctors database."""

SPECIALTIES = {
    "internista": {
        "name": "Internista",
        "doctors": ["dr Anna Kowalska", "dr Piotr Nowak"],
        "description": "choroby wewnętrzne, ogólne dolegliwości",
    },
    "kardiolog": {
        "name": "Kardiolog",
        "doctors": ["dr Maria Wiśniewska", "dr Jan Zieliński"],
        "description": "serce, układ krążenia, ciśnienie",
    },
    "neurolog": {
        "name": "Neurolog",
        "doctors": ["dr Katarzyna Lewandowska"],
        "description": "bóle głowy, zawroty, układ nerwowy",
    },
    "ortopeda": {
        "name": "Ortopeda",
        "doctors": ["dr Tomasz Kamiński", "dr Agnieszka Dąbrowska"],
        "description": "kości, stawy, urazy, ból pleców",
    },
    "dermatolog": {
        "name": "Dermatolog",
        "doctors": ["dr Ewa Szymańska"],
        "description": "skóra, wysypki, zmiany skórne",
    },
    "ginekolog": {
        "name": "Ginekolog",
        "doctors": ["dr Magdalena Jankowska", "dr Aleksandra Wójcik"],
        "description": "zdrowie kobiet, ciąża",
    },
    "pediatra": {
        "name": "Pediatra",
        "doctors": ["dr Michał Kowalczyk", "dr Barbara Mazur"],
        "description": "choroby dzieci",
    },
    "psychiatra": {
        "name": "Psychiatra",
        "doctors": ["dr Robert Krawczyk"],
        "description": "zdrowie psychiczne, lęki, depresja",
    },
    "laryngolog": {
        "name": "Laryngolog (ENT)",
        "doctors": ["dr Joanna Pawlak"],
        "description": "ucho, nos, gardło",
    },
    "okulista": {
        "name": "Okulista",
        "doctors": ["dr Paweł Michalski"],
        "description": "oczy, wzrok",
    },
}


def get_specialties_list() -> str:
    """Get formatted list of specialties for LLM prompt."""
    lines = []
    for key, spec in SPECIALTIES.items():
        lines.append(f"- {spec['name']}: {spec['description']} (lekarze: {', '.join(spec['doctors'])})")
    return "\n".join(lines)


def get_specialty_by_name(name: str) -> dict | None:
    """Find specialty by key or name (case-insensitive)."""
    name_lower = name.lower()
    if name_lower in SPECIALTIES:
        return SPECIALTIES[name_lower]
    for key, spec in SPECIALTIES.items():
        if name_lower in spec["name"].lower() or spec["name"].lower() in name_lower:
            return spec
    return None
