# Facturación y pagos en Nubira

> Nubira es una empresa ficticia creada únicamente con fines demostrativos para este proyecto de portafolio. Todas las políticas descritas a continuación son ficticias.

Este documento explica cómo funciona la facturación y el cobro de las suscripciones de Nubira. Para conocer las políticas de cancelación y reembolso, consulta `04_cancelaciones_y_reembolsos.md`.

## Métodos de pago aceptados

Nubira acepta los siguientes métodos de pago:

- Tarjetas de crédito y débito: Visa, Mastercard y American Express.
- PayPal.

## Facturación mensual vs. anual

- **Facturación mensual**: el cobro se realiza el mismo día calendario de cada mes en que se inició la suscripción. Si un mes no tiene ese día (por ejemplo, el día 31), el cobro se realiza el último día del mes.
- **Facturación anual**: el cobro se realiza una vez al año, en el aniversario de la fecha de inicio de la suscripción. La facturación anual incluye un 20 % de descuento respecto al precio mensual (ver `02_planes_y_precios.md`).

## Cuándo ocurren los cobros

El primer cobro ocurre al finalizar el período de prueba gratuita de 14 días, o de inmediato si el usuario decide contratar un plan sin usar la prueba gratuita. Los cobros posteriores siguen el ciclo mensual o anual elegido.

## Comportamiento ante un pago fallido

Si un cobro no puede procesarse:

1. Nubira reintenta el cobro automáticamente a los 3 días del primer intento fallido.
2. Se realiza un segundo reintento a los 7 días del primer intento fallido.
3. Si el pago sigue sin procesarse después del segundo reintento, la cuenta pasa a un estado de **"restringido"**, en el que solo se puede consultar información existente en modo de solo lectura, sin crear ni editar proyectos o tareas.
4. Si el pago no se regulariza dentro de los **15 días** posteriores al primer intento fallido, la suscripción se cancela automáticamente.

## Facturas y recibos

Después de cada cobro exitoso, Nubira genera automáticamente una factura en PDF y la envía por correo electrónico a la dirección de facturación registrada. Las facturas también quedan disponibles para su descarga en **Configuración > Facturación**.

## Cambiar el método de pago

El método de pago puede actualizarse en cualquier momento desde **Configuración > Facturación**. El nuevo método se utilizará a partir del siguiente cobro programado.

## Cambios de plan: upgrades

Al subir de plan (por ejemplo, de Inicial a Profesional), el cambio se aplica de inmediato y se cobra de forma prorrateada la diferencia correspondiente al resto del período de facturación actual.

## Cambios de plan: downgrades

Al bajar de plan (por ejemplo, de Empresa a Profesional), el cambio se aplica al inicio del siguiente período de facturación. No se realiza ningún reembolso por el período actual ya pagado con el plan anterior.

## Impuestos

Los precios publicados por Nubira no incluyen impuestos. Los impuestos aplicables se calculan y agregan en el momento del pago según el país o región de facturación del cliente, cuando corresponda según la legislación local.

## Moneda

Todos los precios de Nubira están expresados y se cobran en dólares estadounidenses (USD), independientemente del país del cliente.
