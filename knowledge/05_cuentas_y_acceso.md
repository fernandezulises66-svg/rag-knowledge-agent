# Cuentas y acceso en Nubira

> Nubira es una empresa ficticia creada únicamente con fines demostrativos para este proyecto de portafolio. Todas las políticas descritas a continuación son ficticias.

## Creación de cuenta

Una cuenta en Nubira puede crearse de dos maneras: registrándose con un correo electrónico de trabajo y una contraseña, o aceptando una invitación enviada por el administrador de un espacio de trabajo existente.

## Verificación de correo electrónico

Al registrarse, Nubira envía un enlace de verificación al correo indicado. Ese enlace es válido durante **24 horas**. Si expira, se puede solicitar el reenvío del correo de verificación desde la pantalla de inicio de sesión.

## Requisitos de contraseña

Las contraseñas de Nubira deben tener:

- un mínimo de **10 caracteres**,
- al menos una letra mayúscula,
- al menos un número.

## Recuperación de contraseña

Desde la pantalla de inicio de sesión, la opción "¿Olvidaste tu contraseña?" envía un correo con un enlace para restablecerla. Ese enlace es válido durante **1 hora** desde su emisión. Si expira, se puede solicitar uno nuevo repitiendo el proceso.

## Cambiar el correo electrónico de la cuenta

El correo electrónico asociado a una cuenta puede cambiarse desde **Configuración > Perfil**. El nuevo correo debe verificarse antes de que el cambio se aplique por completo.

## Autenticación de dos factores (2FA)

La disponibilidad de 2FA depende del plan contratado:

- **Plan Inicial**: no disponible.
- **Plan Profesional**: disponible y opcional para cada usuario de forma individual.
- **Plan Empresa**: disponible; los administradores de la organización pueden exigirla para todos los miembros de la organización. No es obligatoria de forma predeterminada.

Nubira utiliza códigos de un solo uso (TOTP) generados por una aplicación de autenticación.

## Invitaciones a nuevos usuarios

Un administrador puede invitar a nuevos usuarios por correo electrónico desde **Miembros del equipo > Invitar**. El enlace de invitación es válido durante **7 días**; si expira, el administrador puede reenviarlo.

## Eliminar usuarios

Un administrador puede eliminar a un usuario del espacio de trabajo en cualquier momento desde **Miembros del equipo**. Al eliminarlo, el usuario pierde el acceso de inmediato. Las tareas que tenía asignadas no se reasignan automáticamente; deben reasignarse manualmente a otro miembro del equipo.

## Propiedad de la cuenta

La persona que crea un espacio de trabajo se convierte en su **propietario** de forma predeterminada. La propiedad puede transferirse a otro administrador del mismo espacio de trabajo desde **Configuración > Miembros**. Si el propietario es el único administrador, no puede eliminar su propia cuenta del espacio de trabajo sin transferir antes la propiedad a otra persona.

## Qué ocurre cuando un empleado deja la empresa

Cuando un empleado deja la empresa, un administrador debe eliminar su acceso desde "Miembros del equipo". Las tareas asignadas a esa persona no se reasignan automáticamente, por lo que el administrador debe reasignarlas manualmente a otro miembro del equipo.

## Comportamiento de las sesiones

Una sesión iniciada permanece activa hasta **30 días de inactividad**, después de los cuales se solicitará iniciar sesión nuevamente. Es posible cerrar sesión en todos los dispositivos a la vez desde **Configuración > Seguridad > Cerrar todas las sesiones**.
