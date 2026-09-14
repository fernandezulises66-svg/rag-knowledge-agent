"""Grounded-answer evaluation dataset for the RAG Knowledge Agent.

Reuses the exact same 20 questions, case IDs, and answerability labels
already vetted in `evals/retrieval_cases.py`, so retrieval quality and
final grounded-answer quality can be compared directly on the same
benchmark. `required_fact_groups` and `acceptable_citations` were built
by inspecting the real knowledge-base text and the already-audited
`relevant_targets` for each corresponding retrieval case -- never
inferred from model output. See the per-case comments below for the
exact source sentence each fact group is grounded in.

Language note: this benchmark checks factual grounding, answerability,
citations, and controlled refusal only. It does not attempt to detect
or verify response language (Spanish vs. English); that was verified
manually during the RAG agent smoke test.
"""

from dataclasses import dataclass

from evals.retrieval_cases import RETRIEVAL_EVAL_CASES, RetrievalTarget


@dataclass(frozen=True)
class AnswerEvalCase:
    """A single grounded-answer evaluation case.

    For answerable cases, `required_fact_groups` lists the factual
    concepts a correct answer must mention (each inner tuple is one
    concept; any alternative phrasing within it counts), and
    `acceptable_citations` lists the vetted evidence locations from the
    corresponding retrieval case. Unsupported cases use
    `required_fact_groups=()` and `acceptable_citations=()`.
    """

    case_id: str
    question: str
    answerable: bool
    required_fact_groups: tuple[tuple[str, ...], ...]
    acceptable_citations: tuple[RetrievalTarget, ...]


# Required fact groups per answerable case_id, grounded directly in the
# current knowledge-base text (see comments). Case IDs absent from this
# mapping (all 4 unsupported diagnostic cases) get `()` automatically.
_REQUIRED_FACT_GROUPS_BY_CASE_ID: dict[str, tuple[tuple[str, ...], ...]] = {
    # 02_planes_y_precios.md / "Plan Inicial":
    # "Precio mensual: 10 USD por usuario al mes."
    "pricing-inicial-monthly": (
        ("10 usd", "usd 10", "$10"),
        ("mes", "mensual", "month", "monthly"),
    ),
    # 02_planes_y_precios.md (per-plan sections) / 03_facturacion_y_pagos.md
    # "Facturación mensual vs. anual": "...un 20 % de descuento..."
    "annual-discount": (("20%", "20 %", "20 por ciento"),),
    # 02_planes_y_precios.md / "Plan Empresa": "Mínimo de usuarios: 10."
    "plan-empresa-min-users": (
        ("10 usuarios", "mínimo de 10", "minimum of 10", "at least 10"),
    ),
    # 03_facturacion_y_pagos.md / "Métodos de pago aceptados":
    # "Visa, Mastercard y American Express" + "PayPal."
    "payment-methods": (
        ("visa",),
        ("mastercard", "master card"),
        ("american express", "amex"),
        ("paypal",),
    ),
    # 03_facturacion_y_pagos.md / "Moneda":
    # "...expresados y se cobran en dólares estadounidenses (USD)."
    "billing-currency": (("usd", "dólares estadounidenses", "us dollars", "dollars"),),
    # 04_cancelaciones_y_reembolsos.md / "Acceso después de cancelar":
    # "El acceso no se pierde de inmediato" / continues until period ends.
    # Alternatives below also cover natural paraphrases observed in real
    # generated answers, e.g. "mantendrás el acceso hasta el final del
    # período que ya hayas pagado" -- still the same documented fact,
    # just phrased with "mantener" instead of "continuar"/"no perder".
    "cancel-access-immediate": (
        (
            "no se pierde de inmediato",
            "no pierdes el acceso de inmediato",
            "no pierde el acceso de inmediato",
            "continúa hasta el final",
            "acceso continúa",
            "mantendrás el acceso",
            "mantienes el acceso",
            "hasta el final del período",
            "not lose access immediately",
            "access continues",
            "keep access until the end of the",
        ),
    ),
    # 04_cancelaciones_y_reembolsos.md / "Elegibilidad de reembolso":
    # "...dentro de los 14 días naturales posteriores a ese primer cobro."
    "refund-annual-eligibility": (("14 días", "14 dias", "14 days"),),
    # 04_cancelaciones_y_reembolsos.md / "Cargos duplicados o erróneos":
    # "...dentro de los 60 días posteriores a la fecha del cargo."
    "duplicate-charge": (("60 días", "60 dias", "60 days"),),
    # 05_cuentas_y_acceso.md / "Recuperación de contraseña":
    # "Ese enlace es válido durante 1 hora desde su emisión."
    "password-reset-link-validity": (("1 hora", "una hora", "1 hour", "one hour"),),
    # 05_cuentas_y_acceso.md / "Invitaciones a nuevos usuarios":
    # "El enlace de invitación es válido durante 7 días."
    "invitation-expiration": (("7 días", "7 dias", "7 days"),),
    # 05_cuentas_y_acceso.md / "Autenticación de dos factores (2FA)":
    # available for Plan Profesional and Plan Empresa (not Plan Inicial).
    "2fa-availability": (("profesional",), ("empresa",)),
    # 06_seguridad_y_privacidad.md / "Retención de datos después del
    # cierre de cuenta": "...el período de retención...es de 30 días."
    "data-retention-after-closure": (("30 días", "30 dias", "30 days"),),
    # 06_seguridad_y_privacidad.md / "Exportación de datos":
    # "...en formato CSV..."
    "data-export": (("csv",),),
    # 07_soporte.md / "Horario de atención":
    # "...lunes a viernes, de 8:00 a 20:00 (UTC-5)."
    "support-hours": (
        ("lunes a viernes", "monday to friday", "monday through friday"),
        ("8:00 a 20:00", "8 a 20", "8:00 to 20:00", "8:00-20:00", "8am to 8pm"),
    ),
    # 07_soporte.md / "Tiempos de primera respuesta por plan":
    # Profesional row: "8 horas hábiles." "Horas hábiles" is standard
    # Spanish for "business hours", so "8 business hours" is a correct
    # translation, not a paraphrase that loosens the numeric requirement
    # (the word "business"/"hábiles" sitting between "8" and "hours"/
    # "horas" broke the original plain-substring alternatives).
    "support-response-time-professional": (
        ("8 horas", "8 horas hábiles", "8 hours", "8 business hours", "eight hours", "eight business hours"),
    ),
    # 01_descripcion_producto.md / "Funciones que Nubira NO ofrece
    # actualmente": lists "Nómina..." and "Contabilidad..." as NOT offered.
    "feature-gap-accounting-payroll": (
        (
            "no ofrece",
            "no incluye",
            "no cuenta con",
            "does not offer",
            "does not include",
            "no sirve para",
        ),
        ("contabilidad", "nómina", "nomina", "payroll", "accounting"),
    ),
}


ANSWER_EVAL_CASES: tuple[AnswerEvalCase, ...] = tuple(
    AnswerEvalCase(
        case_id=retrieval_case.case_id,
        question=retrieval_case.question,
        answerable=retrieval_case.answerable,
        required_fact_groups=_REQUIRED_FACT_GROUPS_BY_CASE_ID.get(retrieval_case.case_id, ()),
        acceptable_citations=retrieval_case.relevant_targets,
    )
    for retrieval_case in RETRIEVAL_EVAL_CASES
)
