// Verifica si el navegador soporta acceso a la cámara
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Pide acceso a la cámara
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            // Muestra el stream de video en un elemento <video>
            const video = document.getElementById('video');
            video.srcObject = stream;
            video.play();
        })
        .catch(function (error) {
            console.error("Error al acceder a la cámara: ", error);
        });
} else {
    alert("Tu navegador no soporta acceso a la cámara.");
}
