from typing import Final

SUCCESSFUL_DEVELOPMENT_TYPES: Final[list[str]] = [
    "Under Construction",
    "Operational",
]

UNSUCCESSFUL_DEVELOPMENT_TYPES: Final[list[str]] = [
    "Decommissioned",
    "Planning Permission Expired",
    "Application Refused",
    "Abandoned",
    "Application Withdrawn",
    "Appeal Withdrawn",
]

NEUTRAL_DEVELOPMENT_TYPES: Final[list[str]] = [
    "Awaiting Construction",
    "Application Submitted",
    "Revised",
    "No Application Required",
]

CANCELLED_DEVELOPMENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Appeal Refused",
        "Appeal Withdrawn",
        "Planning Permission Expired",
        "Application Withdrawn",
        "Abandoned",
        "Application Refused",
    }
)

DEVELOPMENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Appeal Refused",
        "Appeal Withdrawn",
        "Appeal Lodged",
        "Planning Permission Expired",
        "Revised",
        "Application Withdrawn",
        "Decommissioned",
        "Under Construction",
        "Awaiting Construction",
        "Abandoned",
        "Application Refused",
        "Operational",
        "Application Submitted",
        "No Application Required",
    }
)

TECHNOLOGY_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Solar Photovoltaics",
        "Compressed Air Energy Storage",
        "Sewage Sludge Digestion",
        "Wind Onshore",
        "Wind Offshore",
        "Biomass (dedicated)",
        "Biomass (co-firing)",
        "Shoreline Wave",
        "Tidal Stream",
        "Tidal Lagoon",
        "Battery",
        "Flywheels",
        "Liquid Air Energy Storage",
        "Fuel Cell (Hydrogen)",
        "Hydrogen",
        "Advanced Conversion Technologies",
        "Air Source Heat Pumps",
        "Geothermal",
        "Hot Dry Rocks (HDR)",
        "Landfill Gas",
        "Small Hydro",
        "Large Hydro",
        "Pumped Storage Hydroelectricity",
        "EfW Incineration",
        "Anaerobic Digestion",
        "Unknown",
    }
)

DATETIME_COLS: Final[list[str]] = [
    "Record Last Updated (dd/mm/yyyy)",
    "Planning Application Submitted",
    "Planning Application Withdrawn",
    "Planning Permission Refused",
    "Appeal Lodged",
    "Appeal Withdrawn",
    "Appeal Refused",
    "Appeal Granted",
    "Planning Permission Granted",
    "Secretary of State - Intervened",
    "Secretary of State - Refusal",
    "Secretary of State - Granted",
    "Planning Permission Expired",
    "Under Construction",
    "Operational",
]
