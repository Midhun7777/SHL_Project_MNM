"""
Role-aware slug packs — context-engineering layer on top of semantic retrieval.

Each rule maps conversational triggers to catalog slugs validated at build time.
This is NOT a bypass of the catalog: slugs must exist in catalog.json.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleRule:
    triggers: tuple[str, ...]
    slugs: tuple[str, ...]


# Ordered most-specific-first; first matching rule contributes its slug pack.
ROLE_RULES: tuple[RoleRule, ...] = (
    RoleRule(
        ("senior leadership", "cxo", "director", "15 years", "leadership benchmark", "selection"),
        (
            "occupational-personality-questionnaire-opq32r",
            "opq-universal-competency-report-2-0",
            "opq-leadership-report",
        ),
    ),
    RoleRule(
        ("rust", "networking infrastructure", "linux programming"),
        (
            "smart-interview-live-coding",
            "linux-programming-general",
            "networking-and-implementation-new",
            "shl-verify-interactive-g",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
    RoleRule(
        ("contact centre", "contact center", "call center", "inbound calls"),
        (
            "svar-spoken-english-us-new",
            "contact-center-call-simulation-new",
            "entry-level-customer-serv-retail-and-contact-center",
            "customer-service-phone-simulation",
        ),
    ),
    RoleRule(
        ("financial analyst", "numerical reasoning", "finance knowledge", "graduate analyst"),
        (
            "shl-verify-interactive-numerical-reasoning",
            "financial-accounting-new",
            "basic-statistics-new",
            "graduate-scenarios",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
    RoleRule(
        ("re-skill", "talent audit", "sales organization", "sales organisation"),
        (
            "global-skills-assessment",
            "global-skills-development-report",
            "occupational-personality-questionnaire-opq32r",
            "opq-mq-sales-report",
            "salestransformationreport2-0-individualcontributor",
        ),
    ),
    RoleRule(
        ("plant operator", "chemical facility", "safety-critical", "safety & dependability", "8.0 bundle"),
        (
            "safety-and-dependability-focus-8-0",
            "workplace-health-and-safety-new",
            "dependability-and-safety-instrument-dsi",
        ),
    ),
    RoleRule(
        ("healthcare admin", "hipaa", "patient records", "south texas", "bilingual"),
        (
            "hipaa-security",
            "medical-terminology-new",
            "microsoft-word-365-essentials-new",
            "dependability-and-safety-instrument-dsi",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
    RoleRule(
        ("admin assistant", "excel and word", "ms excel", "microsoft excel"),
        (
            "microsoft-excel-365-new",
            "microsoft-word-365-new",
            "ms-excel-new",
            "ms-word-new",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
    RoleRule(
        ("full-stack", "full stack", "core java", "spring", "microservice"),
        (
            "core-java-advanced-level-new",
            "spring-new",
            "sql-new",
            "amazon-web-services-aws-development-new",
            "docker-new",
            "shl-verify-interactive-g",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
    RoleRule(
        ("management trainee", "graduate trainee", "cognitive, personality, and situational"),
        (
            "shl-verify-interactive-g",
            "graduate-scenarios",
            "occupational-personality-questionnaire-opq32r",
        ),
    ),
)


def slugs_for_query(query: str) -> list[str]:
    """Return deduplicated catalog slugs whose role rule matched the query."""
    q = query.lower()
    out: list[str] = []
    seen: set[str] = set()
    for rule in ROLE_RULES:
        if any(t in q for t in rule.triggers):
            for slug in rule.slugs:
                if slug not in seen:
                    out.append(slug)
                    seen.add(slug)
    return out
