# Solución de problemas comunes en Nubira

> Nubira es una empresa ficticia creada únicamente con fines demostrativos para este proyecto de portafolio.

Esta guía cubre procedimientos prácticos para los problemas más comunes reportados por los usuarios de Nubira. Si un problema persiste después de seguir estos pasos, debe escalarse al equipo de soporte (ver `07_soporte.md` para canales y horarios).

## No puedo iniciar sesión

1. Verifica que el correo electrónico y la contraseña se escriban correctamente, sin espacios adicionales.
2. Usa la opción "¿Olvidaste tu contraseña?" para restablecerla (ver `05_cuentas_y_acceso.md`).
3. Verifica que tu cuenta no esté en estado "restringido" por un pago fallido (ver `03_facturacion_y_pagos.md`); en ese caso, actualiza el método de pago.
4. Comprueba tu conexión a internet y prueba con otro navegador compatible.
5. **Escalar a soporte** si, tras restablecer la contraseña y confirmar que la cuenta no está restringida, sigues sin poder acceder.

## No recibí el correo de restablecimiento de contraseña

1. Revisa la carpeta de spam o correo no deseado.
2. Confirma que estás usando el mismo correo con el que se registró la cuenta.
3. Espera unos minutos: el envío puede tardar debido a filtros de los proveedores de correo.
4. Solicita un nuevo enlace, ya que el anterior pudo haber expirado (los enlaces son válidos por 1 hora).
5. **Escalar a soporte** si después de varios intentos el correo nunca llega.

## No recibí la invitación al equipo

1. Pide al administrador que confirme que el correo de invitación se escribió correctamente.
2. Revisa la carpeta de spam o correo no deseado.
3. Ten en cuenta que las invitaciones expiran a los 7 días; pide al administrador que reenvíe la invitación si ya pasó ese plazo.
4. **Escalar a soporte** si el problema continúa después de reenviar la invitación.

## Falla la carga de un archivo adjunto

1. Verifica que el archivo no supere el límite de 25 MB por archivo adjunto.
2. Comprueba tu conexión a internet.
3. Intenta con otro navegador compatible o borra la caché del navegador actual.
4. Verifica que el espacio de trabajo no haya alcanzado el límite de almacenamiento de tu plan (ver `02_planes_y_precios.md`).
5. **Escalar a soporte** si el archivo cumple con el límite de tamaño y el espacio de trabajo tiene almacenamiento disponible, pero la carga sigue fallando.

## No recibo notificaciones

1. Revisa las preferencias de notificación en **Configuración > Notificaciones** para confirmar que están activadas.
2. Si son notificaciones por correo, revisa la carpeta de spam.
3. Si son notificaciones dentro de la aplicación, confirma que el navegador tiene permitidas las notificaciones para el sitio de Nubira.
4. **Escalar a soporte** si las preferencias están correctamente configuradas y las notificaciones aún no llegan.

## Una automatización no se ejecutó

1. Verifica que la automatización esté activa (habilitada) en el panel del proyecto.
2. Revisa que la condición del disparador realmente se haya cumplido (por ejemplo, que la tarea se haya movido a la columna correcta).
3. Comprueba si el espacio de trabajo alcanzó el límite mensual de ejecuciones de automatización de su plan (ver `02_planes_y_precios.md`); si se alcanzó, las automatizaciones adicionales no se ejecutan hasta el siguiente ciclo.
4. Revisa el historial de la automatización para ver si quedó registrado algún error.
5. **Escalar a soporte** si la automatización está activa, no se ha alcanzado el límite mensual y el disparador sí se cumplió, pero la acción no se ejecutó.

## Criterio general de escalación

Cualquier problema que no se resuelva con los pasos anteriores debe reportarse al equipo de soporte a través de los canales indicados en `07_soporte.md`, incluyendo la información solicitada (espacio de trabajo, pasos para reproducir, capturas de pantalla y navegador utilizado) para agilizar el diagnóstico.
