"""Tests pour les outils de l'agent."""


def test_all_tools_list():
    """Vérifie que ALL_TOOLS contient les 6 outils."""
    from src.agents.tools import ALL_TOOLS

    assert len(ALL_TOOLS) == 6


def test_tool_names():
    """Vérifie les noms des outils."""
    from src.agents.tools import ALL_TOOLS

    noms = [t.name for t in ALL_TOOLS]
    attendus = [
        "get_weather",
        "get_historical_weather",
        "get_forecast",
        "web_search",
        "calculator",
        "search_corpus",
    ]
    for nom in attendus:
        assert nom in noms, f"Outil '{nom}' manquant dans ALL_TOOLS"


def test_calculator_basic():
    """Vérifie que la calculatrice fonctionne."""
    from src.agents.tools import calculator

    result = calculator.invoke({"expression": "3+7*2"})
    assert "17" in result


def test_calculator_sqrt():
    """Vérifie les fonctions mathématiques."""
    from src.agents.tools import calculator

    result = calculator.invoke({"expression": "sqrt(144)"})
    assert "12" in result


def test_calculator_division_zero():
    """Vérifie la gestion de la division par zéro."""
    from src.agents.tools import calculator

    result = calculator.invoke({"expression": "1/0"})
    assert "zéro" in result.lower() or "zero" in result.lower()


def test_geocode():
    """Vérifie le géocodage d'une ville."""
    from src.agents.tools import _geocode

    result = _geocode("Paris")
    assert result is not None
    lat, lon, name = result
    assert 48 < lat < 49
    assert 2 < lon < 3


def test_geocode_unknown_city():
    """Vérifie le géocodage d'une ville inexistante."""
    from src.agents.tools import _geocode

    result = _geocode("VilleQuiNExistePas12345")
    assert result is None


def test_wmo_codes():
    """Vérifie que les codes WMO sont définis."""
    from src.agents.tools import _WMO_CODES

    assert 0 in _WMO_CODES
    assert 95 in _WMO_CODES
    assert "dégagé" in _WMO_CODES[0].lower() or "Ciel" in _WMO_CODES[0]
