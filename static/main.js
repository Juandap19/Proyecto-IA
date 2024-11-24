const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const serverURL = '/process_frame';
const activityDisplay = document.getElementById('activity-display'); // Elemento para mostrar actividad

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.play();

            // Enviar frames al servidor cada 200 ms
            setInterval(() => {
                sendFrameToServer(video);
            }, 200);
        })
        .catch((error) => {
            console.error("Error al acceder a la cámara: ", error);
        });
}

async function sendFrameToServer(video) {
    // Dibujar el frame del video en un canvas
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convertir el canvas a Base64
    const frame = canvas.toDataURL('image/jpeg');

    // Enviar el frame al servidor
    try {
        const response = await fetch(serverURL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame }),
        });

        const data = await response.json();

        // Actualizar el elemento con la actividad detectada
        if (data.activity) {
            activityDisplay.textContent = `Actividad detectada: ${data.activity}`;
        } else {
            activityDisplay.textContent = "No se detectó actividad.";
        }

    } catch (error) {
        console.error('Error enviando el frame:', error);
        activityDisplay.textContent = "Error al detectar actividad.";
    }
}
