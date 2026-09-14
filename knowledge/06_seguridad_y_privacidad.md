# Seguridad y privacidad en Nubira

> Nubira es una empresa ficticia creada únicamente con fines demostrativos para este proyecto de portafolio. Todas las políticas descritas a continuación son ficticias y no deben interpretarse como afirmaciones reales de cumplimiento normativo.

## Cifrado en tránsito

Toda la información que viaja entre el navegador del usuario y los servidores de Nubira está cifrada mediante TLS 1.2 o superior.

## Cifrado en reposo

Los datos almacenados por Nubira se cifran utilizando el algoritmo AES-256.

## Controles de acceso

El acceso dentro de un espacio de trabajo se gestiona mediante roles:

- **Administrador** y **Miembro** están disponibles en todos los planes.
- Los **roles personalizados**, con permisos configurables por sección, están disponibles únicamente en los planes Profesional y Empresa (ver `02_planes_y_precios.md`).

## Copias de seguridad

Nubira realiza copias de seguridad automáticas de los datos de cada espacio de trabajo con frecuencia diaria. Estas copias se conservan durante **30 días**.

## Seguridad de la cuenta

Además de los requisitos de contraseña y la autenticación de dos factores descritos en `05_cuentas_y_acceso.md`, Nubira notifica por correo electrónico cuando se detecta un inicio de sesión desde un dispositivo no reconocido previamente.

## Datos del usuario y eliminación de datos

Cuando una cuenta se cierra o se cancela por completo, los datos del espacio de trabajo se conservan durante **30 días** por si el propietario decide reactivar la cuenta. Transcurrido ese plazo, los datos se eliminan de forma permanente e irreversible de los sistemas de Nubira.

## Retención de datos después del cierre de cuenta

Como se indica arriba, el período de retención tras el cierre de una cuenta es de 30 días. Durante ese período, los datos no son accesibles públicamente y solo pueden restaurarse mediante una solicitud de reactivación.

## Exportación de datos

Los usuarios pueden exportar los proyectos y tareas de su espacio de trabajo en formato CSV en cualquier momento mientras la cuenta esté activa, y también durante los 30 días posteriores al cierre de la cuenta, antes de la eliminación definitiva de los datos.

## Contacto de privacidad

Las consultas relacionadas con privacidad y protección de datos pueden dirigirse a **privacidad@nubira.io**.

## Sobre certificaciones de cumplimiento

Nubira seguirá prácticas internas de seguridad como las descritas en este documento. Nubira **no** cuenta actualmente con certificaciones formales de terceros como SOC 2, ISO 27001, HIPAA o certificaciones de cumplimiento de GDPR, y no debe afirmarse que las posee.

## Funciones de seguridad no disponibles actualmente

Nubira no ofrece actualmente integración de inicio de sesión único (SSO/SAML) para autenticación empresarial. Esta función no está disponible en ningún plan.
