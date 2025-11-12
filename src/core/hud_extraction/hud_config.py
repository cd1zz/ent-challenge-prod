"""
Centralized HUD Field Configuration
Single source of truth for HUD fields extracted across all methods (GPT-4V, Tesseract OCR, etc.)

Configuration is loaded from .env file. Set HUD_FIELDS environment variable with comma-separated field names.
Example: HUD_FIELDS=compass_heading,game_status,player_rank,equipped_accessories,weapon_and_ammo,player_health,team_health
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


def get_hud_fields() -> List[str]:
    """
    Get list of HUD fields to extract from .env file.

    Environment variable format (in .env):
        HUD_FIELDS=compass_heading,game_status,player_rank,equipped_accessories,weapon_and_ammo,player_health,team_health

    Returns:
        List of field names to extract

    Raises:
        ValueError: If HUD_FIELDS is not set in .env
    """
    fields_str = os.getenv('HUD_FIELDS')
    if not fields_str:
        raise ValueError(
            "HUD_FIELDS not found in .env file. Please add:\n"
            "HUD_FIELDS=compass_heading,game_status,player_rank,equipped_accessories,weapon_and_ammo,player_health,team_health"
        )
    return [f.strip() for f in fields_str.split(',') if f.strip()]


def get_gpt4v_field_descriptions() -> Dict[str, str]:
    """
    Get GPT-4V field descriptions for prompt engineering.
    Maps field names to natural language descriptions for GPT-4V.

    Returns:
        Dictionary mapping field names to descriptions
    """
    descriptions = {
        'compass_heading': 'Compass direction at top center (format: "N", "NE", "330", etc.)',
        'game_status': 'Game status info at top right containing: Kills (player kills count), Assists (assisted kills), Teams (teams remaining), Soldiers (individual soldiers remaining). Format: "Kills: 3, Teams: 18, Soldiers: 52"',
        'player_rank': 'Player rank and class at right side (format: "Lv 32 Firearms Expert")',
        'equipped_accessories': 'Current player extra gear displayed in a 3x3 box at lower right (tracers, enhanced aim, etc.). List all visible accessories.',
        'weapon_and_ammo': 'Current weapon and ammo status at lower center right (format: "MG 30.0 | 30/120" where 30 is current magazine, 120 is reserve)',
        'player_health': 'Current player health at lower center (format: "110/110" - current/max)',
        'team_health': 'Team member names and health status in a stack at lower left. Format: list of "PlayerName: 85/100"'
    }

    # Only return descriptions for enabled fields
    enabled_fields = get_hud_fields()
    return {field: descriptions[field] for field in enabled_fields if field in descriptions}


def get_ocr_field_types() -> Dict[str, str]:
    """
    Get OCR field types for Tesseract configuration.
    Maps field names to data types (number, text, currency).

    Returns:
        Dictionary mapping field names to types
    """
    types = {
        'compass_heading': 'text',
        'game_status': 'text',
        'player_rank': 'text',
        'equipped_accessories': 'text',
        'weapon_and_ammo': 'text',
        'player_health': 'number',
        'team_health': 'text'
    }

    # Only return types for enabled fields
    enabled_fields = get_hud_fields()
    return {field: types[field] for field in enabled_fields if field in types}
