"""Retrieval evaluation dataset for the RAG Knowledge Agent.

Defines the case model and a small, hand-curated set of evaluation
cases built directly against the real `knowledge/` documents and the
exact sections the current deterministic chunker produces for them.
See `tests/test_retrieval_evals.py` for a consistency check that keeps
this dataset from silently going stale if knowledge-base headings
change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalTarget:
    """One acceptable evidence location for an evaluation case.

    A target with a section requires an exact source and section
    match. A target with `section=None` matches any section from that
    source, for the occasional case where more than one section is
    legitimately relevant.
    """

    source: str
    section: str | None


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A single retrieval evaluation case.

    For answerable cases, `relevant_targets` lists one or more
    acceptable evidence locations. Unsupported cases intentionally
    have no evidence in the knowledge base and use `relevant_targets=()`
    — they must never be scored with Hit@K or MRR.
    """

    case_id: str
    question: str
    answerable: bool
    relevant_targets: tuple[RetrievalTarget, ...]


RETRIEVAL_EVAL_CASES: tuple[RetrievalEvalCase, ...] = (
    # --- Answerable cases ---------------------------------------------
    RetrievalEvalCase(
        case_id="pricing-inicial-monthly",
        question="¿Cuál es el precio mensual del plan Inicial por usuario?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(source="02_planes_y_precios.md", section="Plan Inicial"),
            # The comparison table restates the exact monthly price per plan.
            RetrievalTarget(
                source="02_planes_y_precios.md", section="Diferencias clave entre planes"
            ),
            # The FAQ independently restates "Inicial (10 USD/usuario/mes)".
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Cuáles son los planes disponibles y sus precios?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="annual-discount",
        question=(
            "Si pago mi suscripción una vez al año en lugar de cada mes, "
            "¿me sale más barato?"
        ),
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="03_facturacion_y_pagos.md",
                section="Facturación mensual vs. anual",
            ),
            # Each per-plan section explicitly states "un 20 % de descuento
            # frente a la facturación mensual" — direct, independent evidence.
            # (The comparison table only shows raw prices, requiring the
            # reader to compute the percentage themselves, so it is not
            # included as direct evidence.)
            RetrievalTarget(source="02_planes_y_precios.md", section="Plan Inicial"),
            RetrievalTarget(source="02_planes_y_precios.md", section="Plan Profesional"),
            RetrievalTarget(source="02_planes_y_precios.md", section="Plan Empresa"),
            # The FAQ independently states "la facturación anual ... tiene un
            # 20 % de descuento".
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Cuáles son los planes disponibles y sus precios?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="plan-empresa-min-users",
        question="¿Cuántos usuarios como mínimo necesito para contratar el plan Empresa?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(source="02_planes_y_precios.md", section="Plan Empresa"),
            # The comparison table's "Usuarios mínimos" row states Empresa = 10.
            RetrievalTarget(
                source="02_planes_y_precios.md", section="Diferencias clave entre planes"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="payment-methods",
        question="What payment methods does Nubira accept?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="03_facturacion_y_pagos.md", section="Métodos de pago aceptados"
            ),
            # The FAQ independently restates the identical payment-method list.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Qué métodos de pago se aceptan?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="billing-currency",
        question="¿En qué moneda se realizan los cobros de Nubira?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(source="03_facturacion_y_pagos.md", section="Moneda"),
            # The FAQ independently states prices are expressed and charged in USD.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md", section="¿Qué monedas admite Nubira?"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="cancel-access-immediate",
        question="Si cancelo hoy mi suscripción, ¿pierdo el acceso de inmediato?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Acceso después de cancelar",
            ),
            # This section's own Q&A restates the exact question and answer.
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Preguntas comunes resueltas por esta política",
            ),
            # The FAQ independently states access is not lost immediately.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Qué pasa cuando cancelo mi suscripción?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="refund-annual-eligibility",
        question=(
            "¿Puedo pedir que me devuelvan el dinero de un plan anual "
            "que acabo de contratar?"
        ),
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Elegibilidad de reembolso",
            ),
            # This section's own Q&A restates the annual-refund rule verbatim.
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Preguntas comunes resueltas por esta política",
            ),
            # The FAQ independently states the same first-charge/14-day rule.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md", section="¿Puedo obtener un reembolso?"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="duplicate-charge",
        question="Me cobraron dos veces el mismo mes, ¿qué debo hacer?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Cargos duplicados o erróneos",
            ),
            # This section's own Q&A restates the exact question and answer.
            RetrievalTarget(
                source="04_cancelaciones_y_reembolsos.md",
                section="Preguntas comunes resueltas por esta política",
            ),
            # The FAQ independently states duplicate charges are refundable
            # within 60 days.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md", section="¿Puedo obtener un reembolso?"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="password-reset-link-validity",
        question="How long does a password reset link remain valid?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="05_cuentas_y_acceso.md", section="Recuperación de contraseña"
            ),
            # This troubleshooting step independently states "los enlaces son
            # válidos por 1 hora".
            RetrievalTarget(
                source="08_solucion_de_problemas.md",
                section="No recibí el correo de restablecimiento de contraseña",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="invitation-expiration",
        question="¿Cuánto tiempo tengo para aceptar una invitación antes de que caduque?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="05_cuentas_y_acceso.md", section="Invitaciones a nuevos usuarios"
            ),
            # This troubleshooting step independently states "las invitaciones
            # expiran a los 7 días".
            RetrievalTarget(
                source="08_solucion_de_problemas.md",
                section="No recibí la invitación al equipo",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="2fa-availability",
        question="¿En qué planes está disponible la verificación en dos pasos?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="05_cuentas_y_acceso.md",
                section="Autenticación de dos factores (2FA)",
            ),
            # The comparison table's 2FA row restates availability per plan.
            RetrievalTarget(
                source="02_planes_y_precios.md", section="Diferencias clave entre planes"
            ),
            # The FAQ independently restates the same per-plan breakdown.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Nubira tiene autenticación de dos factores (2FA)?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="data-retention-after-closure",
        question=(
            "Si cierro mi cuenta, ¿cuánto tiempo guardan mis datos antes "
            "de borrarlos para siempre?"
        ),
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="06_seguridad_y_privacidad.md",
                section="Retención de datos después del cierre de cuenta",
            ),
            # This section independently states the same 30-day/permanent-
            # deletion outcome.
            RetrievalTarget(
                source="06_seguridad_y_privacidad.md",
                section="Datos del usuario y eliminación de datos",
            ),
            # The FAQ independently restates the same fact.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Qué pasa con mis datos si cierro mi cuenta?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="data-export",
        question="¿Es posible descargar toda la información de mis proyectos y tareas?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="06_seguridad_y_privacidad.md", section="Exportación de datos"
            ),
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Puedo exportar mis datos?",
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="support-hours",
        question="¿En qué horario puedo contactar al equipo de soporte?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(source="07_soporte.md", section="Horario de atención"),
            # The FAQ independently restates the exact support hours.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md", section="¿Cómo contacto a soporte?"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="support-response-time-professional",
        question=(
            "How quickly can I expect a first response from support on "
            "the Professional plan?"
        ),
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="07_soporte.md", section="Tiempos de primera respuesta por plan"
            ),
        ),
    ),
    RetrievalEvalCase(
        case_id="feature-gap-accounting-payroll",
        question="¿Nubira sirve para llevar la contabilidad o la nómina de mi empresa?",
        answerable=True,
        relevant_targets=(
            RetrievalTarget(
                source="01_descripcion_producto.md",
                section="Funciones que Nubira NO ofrece actualmente",
            ),
            # The FAQ independently and directly answers the same question.
            RetrievalTarget(
                source="09_preguntas_frecuentes.md",
                section="¿Nubira ofrece funciones de contabilidad, nómina o CRM?",
            ),
        ),
    ),
    # --- Unsupported diagnostic cases (intentionally no evidence) -----
    RetrievalEvalCase(
        case_id="unsupported-crypto",
        question="¿Nubira acepta pagos con criptomonedas?",
        answerable=False,
        relevant_targets=(),
    ),
    RetrievalEvalCase(
        case_id="unsupported-roadmap",
        question="¿Cuáles son las próximas funciones que Nubira lanzará este año?",
        answerable=False,
        relevant_targets=(),
    ),
    RetrievalEvalCase(
        case_id="unsupported-salaries",
        question="¿Cuánto ganan los empleados de Nubira?",
        answerable=False,
        relevant_targets=(),
    ),
    RetrievalEvalCase(
        case_id="unsupported-office-address",
        question="¿Cuál es la dirección física de las oficinas de Nubira?",
        answerable=False,
        relevant_targets=(),
    ),
)
