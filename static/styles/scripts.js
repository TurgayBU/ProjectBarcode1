const container = document.getElementById('fireworks');
        const fireworks = new Fireworks.default(container, {
            hue: { min: 45, max: 55 },
            delay: { min: 15, max: 30 },
            speed: 2,
            acceleration: 1.05,
            friction: 0.98,
            gravity: 1.5,
            particles: 90,
            trace: 3,
            explosion: 5,
            autoresize: true,
            brightness: { min: 50, max: 80 },
            decay: { min: 0.015, max: 0.03 },
            mouse: { click: false, move: false, max: 2 }
        });
        fireworks.start();

        let cameraOn = false;

        function toggleCamera() {
            const video = document.getElementById('cameraFeed');
            const btn = document.getElementById('cameraToggleBtn');
            const captureButton = document.getElementById('captureButton');

            if (!cameraOn) {
                video.style.display = 'block';
                captureButton.style.display = 'block';
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        video.srcObject = stream;
                    })
                    .catch(err => {
                        alert("Kamera açılamadı.");
                    });
                btn.innerText = 'Stop Camera';
            } else {
                video.style.display = 'none';
                captureButton.style.display = 'none';
                const stream = video.srcObject;
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                video.srcObject = null;
                btn.innerText = 'Start Camera';
            }

            cameraOn = !cameraOn;
        }

        function capturePhoto() {
            const video = document.getElementById('cameraFeed');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL('image/jpeg');
            document.getElementById('imageData').value = imageData;

            // Gönderim formunu göster
            document.getElementById('uploadCameraForm').style.display = 'block';
        }

        function speakProductInfo() {
    const { productName, brandName, origin, productDate, price } = productInfo;

    const productText = `
        Product Information:
        Product Name: ${productName},
        Brand Name: ${brandName},
        Price: ${price} Turkish Lira.
        Origin: ${origin},
        Product Date: ${productDate},
    `;

    const utterance = new SpeechSynthesisUtterance(productText);
    utterance.lang = 'en-US';

    function setVoiceAndSpeak() {
        const voices = window.speechSynthesis.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en'));

        if (englishVoice) {
            utterance.voice = englishVoice;
            speechSynthesis.speak(utterance);
        } else {
            console.warn("İngilizce ses bulunamadı.");
        }
    }

    if (speechSynthesis.getVoices().length === 0) {
        speechSynthesis.onvoiceschanged = setVoiceAndSpeak;
    } else {
        setVoiceAndSpeak();
    }
}

   let currentIndex = 0;

const imageElements = [
    document.getElementById('bg-left'),
    document.getElementById('bg-center'),
    document.getElementById('bg-right')
];

function updateCarousel() {
    imageElements.forEach((img, i) => {
        img.classList.remove('active');
    });

    imageElements[currentIndex % imageElements.length].classList.add('active');
    currentIndex = (currentIndex + 1) % imageElements.length;
}

// İlk başlat
updateCarousel();
setInterval(updateCarousel, 5000);
