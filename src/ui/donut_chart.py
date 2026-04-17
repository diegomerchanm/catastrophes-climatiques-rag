"""
Generateur de donut chart SVG — style professionnel fond blanc.
Toutes les parts sont toujours visibles. Les outils actifs sont en couleur,
les inactifs sont grises. Les pourcentages changent dynamiquement.
"""

import math

# Toutes les categories du systeme (toujours presentes dans le donut)
ALL_CATEGORIES = [
    "RAG",
    "Meteo",
    "Web",
    "Calcul",
    "ML",
    "Scoring",
    "Email",
    "Agent",
    "Chat",
]

# Couleurs actives
TOOL_COLORS = {
    "RAG": "#2563eb",
    "Meteo": "#facc15",
    "Web": "#10b981",
    "Calcul": "#8b5cf6",
    "ML": "#ef4444",
    "Scoring": "#f0abfc",
    "Email": "#06b6d4",
    "Agent": "#f97316",
    "Chat": "#16a34a",
}

# Couleurs inactives — memes couleurs, juste un peu moins saturees
TOOL_COLORS_LIGHT = {
    "RAG": "#3b82f6",
    "Meteo": "#fde047",
    "Web": "#34d399",
    "Calcul": "#a78bfa",
    "ML": "#f87171",
    "Scoring": "#f0abfc",
    "Email": "#22d3ee",
    "Agent": "#fb923c",
    "Chat": "#22c55e",
}

# Mapping outil -> categorie
TOOL_CATEGORIES = {
    "search_corpus": "RAG",
    "get_weather": "Meteo",
    "get_historical_weather": "Meteo",
    "get_forecast": "Meteo",
    "web_search": "Web",
    "calculator": "Calcul",
    "predict_risk": "ML",
    "predict_risk_by_type": "ML",
    "calculer_score_risque": "Scoring",
    "send_email": "Email",
    "send_bulk_email": "Email",
    "schedule_email": "Email",
    "list_corpus": "RAG",
    "__agent__": "Agent",
}


def _identifier_actifs(outils_appeles: list, route: str) -> dict:
    """Identifie quelles categories sont actives et leur poids."""
    actifs = {}
    for outil in outils_appeles:
        cat = TOOL_CATEGORIES.get(outil, "Chat")
        actifs[cat] = actifs.get(cat, 0) + 1

    if not actifs:
        route_map = {"rag": "RAG", "agent": "Agent", "chat": "Chat"}
        actifs[route_map.get(route, "Chat")] = 1

    return actifs


def generer_message_avec_donut(
    answer: str,
    outils_appeles: list,
    route: str,
    sources: list = None,
    tokens_info: str = "",
) -> str:
    """
    Genere le message complet avec donut a gauche et texte a droite.
    Toutes les parts du donut sont toujours visibles.
    Les categories actives sont en couleur, les inactives en gris.
    """
    actifs = _identifier_actifs(outils_appeles, route)
    total_actifs = sum(actifs.values())

    # Construire les parts : actifs prennent leur proportion reelle,
    # inactifs prennent une petite part fixe
    parts = []
    nb_inactifs = len(ALL_CATEGORIES) - len(actifs)
    part_inactive = 5  # 5% par inactif
    part_active_total = 100 - (nb_inactifs * part_inactive)

    for cat in ALL_CATEGORIES:
        if cat in actifs:
            pct = (actifs[cat] / total_actifs) * part_active_total
            parts.append((cat, pct, True))
        else:
            parts.append((cat, part_inactive, False))

    # Generer le SVG (+50%)
    size = 240
    cx, cy = 120, 120
    r, r_inner = 97, 60

    start_angle = -90
    arcs = ""

    for cat, pct, actif in parts:
        angle = (pct / 100) * 360
        end_angle = start_angle + angle
        color = TOOL_COLORS[cat] if actif else TOOL_COLORS_LIGHT[cat]
        opacity = "1"

        if angle >= 359.9:
            mid_r = (r + r_inner) / 2
            arcs += (
                f'<circle cx="{cx}" cy="{cy}" r="{mid_r}" '
                f'fill="none" stroke="{color}" '
                f'stroke-width="{r - r_inner}" opacity="{opacity}"/>'
            )
        else:
            large_arc = 1 if angle > 180 else 0
            x1o = cx + r * math.cos(math.radians(start_angle))
            y1o = cy + r * math.sin(math.radians(start_angle))
            x2o = cx + r * math.cos(math.radians(end_angle))
            y2o = cy + r * math.sin(math.radians(end_angle))
            x1i = cx + r_inner * math.cos(math.radians(end_angle))
            y1i = cy + r_inner * math.sin(math.radians(end_angle))
            x2i = cx + r_inner * math.cos(math.radians(start_angle))
            y2i = cy + r_inner * math.sin(math.radians(start_angle))

            path = (
                f"M {x1o:.1f} {y1o:.1f} "
                f"A {r} {r} 0 {large_arc} 1 {x2o:.1f} {y2o:.1f} "
                f"L {x1i:.1f} {y1i:.1f} "
                f"A {r_inner} {r_inner} 0 {large_arc} 0 "
                f"{x2i:.1f} {y2i:.1f} Z"
            )
            arcs += (
                f'<path d="{path}" fill="{color}" opacity="{opacity}"'
                f' stroke="#ffffff" stroke-width="1.5"/>'
            )

        start_angle = end_angle

    # Texte central
    nb_outils = len(actifs)
    center_text = str(total_actifs) if total_actifs > 1 else list(actifs.keys())[0]
    sub_text = "outils" if total_actifs > 1 else ""

    donut_svg = (
        f'<svg width="{size}" height="{size}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="filter:drop-shadow(0 2px 4px rgba(0,0,0,0.1));">'
        f'<circle cx="{cx}" cy="{cy}" r="{r + 2}" '
        f'fill="none" stroke="#f3f4f6" stroke-width="1"/>'
        f"<g>{arcs}</g>"
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" '
        f'font-size="24" font-weight="bold" fill="#1f2937" '
        f'font-family="Inter, sans-serif">{center_text}</text>'
        f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" '
        f'font-size="10" fill="#6b7280" '
        f'font-family="Inter, sans-serif">{sub_text}</text>'
        f"</svg>"
    )

    # Legende — sous le donut
    legend_items = ""
    for cat, pct, actif in parts:
        color = TOOL_COLORS[cat] if actif else TOOL_COLORS_LIGHT[cat]
        text_color = "#1f2937" if actif else "#6b7280"
        font_weight = "600" if actif else "400"
        real_pct = f"{pct:.0f}%"
        legend_items += (
            f'<span style="display:inline-flex;align-items:center;'
            f"margin-right:10px;margin-bottom:4px;"
            f'font-size:11px;color:{text_color};font-weight:{font_weight};">'
            f'<span style="display:inline-block;width:8px;height:8px;'
            f"border-radius:50%;background:{color};"
            f'margin-right:4px;"></span>'
            f"{cat} {real_pct}</span>"
        )

    legend_html = (
        f'<div style="display:flex;flex-direction:column;'
        f'gap:2px;">{legend_items}</div>'
    )

    # Sources
    sources_html = ""
    if sources:
        sources_html = (
            '<div style="margin-top:10px;padding-top:8px;'
            'border-top:1px solid #374151;font-size:12px;color:#1f2937;">'
            "<b>Sources :</b><br>"
        )
        for src_item in sources[:12]:
            sources_html += f"&bull; {src_item}<br>"
        sources_html += "</div>"

    # Tokens
    tokens_html = ""
    if tokens_info:
        tokens_html = (
            f'<div style="margin-top:6px;font-size:11px;'
            f'color:#374151;font-style:italic;">{tokens_info}</div>'
        )

    # Conversion markdown -> HTML via la lib officielle (robuste tout cas limite)
    import re

    # Normalisation prealable : le LLM concatene parfois "texte.## Titre"
    # -> inserer un saut de ligne avant les headers colles a la ponctuation
    norm = re.sub(r"([.!?;:])\s*(#{1,4}\s)", r"\1\n\n\2", answer)
    norm = re.sub(r"([a-zA-ZéèêàâùûçÉÈÀ])\s*(#{1,4}\s)", r"\1\n\n\2", norm)

    try:
        import markdown as _md

        answer_html = _md.markdown(norm, extensions=["fenced_code", "tables", "nl2br"])
    except ImportError:
        # Fallback regex si lib absente
        answer_html = norm
        answer_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", answer_html)
        answer_html = re.sub(
            r"^[ \t]*##\s+(.+?)\s*$",
            r"<b style='font-size:1.15em'>\1</b>",
            answer_html,
            flags=re.MULTILINE,
        )
        answer_html = answer_html.replace("\n", "<br>")

    # Layout vertical : donut + legende en haut (cote a cote), texte pleine
    # largeur en dessous. Fiable desktop ET mobile : Chainlit wrappe tout
    # dans son propre flex, donc on laisse le texte circuler sans contrainte
    # de colonne. Le donut + legende restent groupes visuellement comme
    # un "avatar + legende" en tete de message.
    html = (
        f'<div style="padding:8px 0;">'
        f'<div style="display:flex;align-items:center;gap:16px;'
        f'margin-bottom:12px;flex-wrap:wrap;">'
        f'<div style="flex-shrink:0;">{donut_svg}</div>'
        f'<div style="flex-shrink:0;">{legend_html}</div>'
        f"</div>"
        f'<div style="color:#1f2937;line-height:1.6;">{answer_html}</div>'
        f"{sources_html}{tokens_html}"
        f"</div>"
    )

    return html
