document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const removeImageBtn = document.getElementById('remove-image-btn');
    const fileLabel = document.getElementById('file-label');
    const uploadForm = document.getElementById('upload-form');
    
    const loadingScreen = document.getElementById('loading');
    const resultScreen = document.getElementById('result');
    const extractedTextDiv = document.getElementById('extracted-text');
    const confidenceScore = document.getElementById('confidence-score');
    const statusBadge = document.getElementById('status-badge');
    const mainElement = document.querySelector('main');

    // Drag and drop handling
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updateImagePreview();
        }
    });

    // Make clicking anywhere in the drop zone open the file dialog
    dropZone.addEventListener('click', (e) => {
        // Don't trigger if they clicked the remove button
        if (e.target.id !== 'remove-image-btn') {
            fileInput.click();
        }
    });

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent clicking the drop-zone
        fileInput.value = ''; // clear the file input
        updateImagePreview();
        mainElement.classList.remove('split-screen');
    });

    fileInput.addEventListener('change', updateImagePreview);

    function updateImagePreview() {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'flex';
                fileLabel.style.display = 'none';
                removeImageBtn.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            previewContainer.style.display = 'none';
            fileLabel.style.display = 'block';
            removeImageBtn.style.display = 'none';
            mainElement.classList.remove('split-screen');
        }
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!fileInput.files[0]) {
            alert('PLEASE PROVIDE IMAGE DATA');
            return;
        }

        // Trigger the slide animation
        mainElement.classList.add('split-screen');

        // Show loading state
        loadingScreen.classList.remove('hidden');
        resultScreen.classList.add('hidden');
        extractedTextDiv.innerHTML = '';
        extractedTextDiv.classList.remove('typing');

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingScreen.classList.add('hidden');
            resultScreen.classList.remove('hidden');

            if (data.status === 'success') {
                confidenceScore.innerText = `CONFIDENCE: ${(data.confidence * 100).toFixed(1)}%`;
                
                if (data.confidence > 0.9) {
                    statusBadge.innerText = 'HIGH_ACCURACY';
                    statusBadge.style.borderColor = '#39ff14';
                    statusBadge.style.color = '#39ff14';
                    statusBadge.style.background = 'rgba(57, 255, 20, 0.2)';
                } else {
                    statusBadge.innerText = 'LOW_ACCURACY_DETECTED';
                    statusBadge.style.borderColor = '#ff00ea';
                    statusBadge.style.color = '#ff00ea';
                    statusBadge.style.background = 'rgba(255, 0, 234, 0.2)';
                }

                typeWriterEffect(data.text);
            } else {
                confidenceScore.innerText = 'ERROR';
                statusBadge.innerText = 'FAILED';
                extractedTextDiv.innerText = `SYSTEM_ERROR: ${data.message}`;
                extractedTextDiv.style.color = '#ff00ea';
            }
        } catch (error) {
            loadingScreen.classList.add('hidden');
            resultScreen.classList.remove('hidden');
            confidenceScore.innerText = 'ERROR';
            statusBadge.innerText = 'FAILED';
            extractedTextDiv.innerText = `NETWORK_ERROR: ${error.message}`;
            extractedTextDiv.style.color = '#ff00ea';
        }
    });

    function typeWriterEffect(text) {
        extractedTextDiv.classList.add('typing');
        let i = 0;
        extractedTextDiv.innerHTML = '';
        
        function type() {
            if (i < text.length) {
                // Handle newlines
                if (text.charAt(i) === '\n') {
                    extractedTextDiv.innerHTML += '<br>';
                } else {
                    extractedTextDiv.innerHTML += text.charAt(i);
                }
                i++;
                setTimeout(type, 15); // typing speed
            } else {
                extractedTextDiv.classList.remove('typing');
            }
        }
        
        type();
    }

    // Matrix Rain Animation
    const canvas = document.getElementById('matrix-canvas');
    const ctx = canvas.getContext('2d');

    let cw = window.innerWidth;
    let ch = window.innerHeight;
    canvas.width = cw;
    canvas.height = ch;

    const katakana = 'アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプエェケセテネヘメレゲゼデベペオォコソトノホモヨョロゴゾドボポヴッン';
    const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const nums = '0123456789';
    const alphabet = katakana + latin + nums;

    const fontSize = 16;
    let columns = cw / fontSize;

    let rainDrops = [];
    for (let x = 0; x < columns; x++) {
        rainDrops[x] = 1;
    }

    const draw = () => {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.font = fontSize + 'px monospace';

        for (let i = 0; i < rainDrops.length; i++) {
            const text = alphabet.charAt(Math.floor(Math.random() * alphabet.length));
            
            // Randomly flash our theme colors
            const randColor = Math.random();
            if (randColor > 0.98) ctx.fillStyle = '#ff00ea'; // Pink
            else if (randColor > 0.95) ctx.fillStyle = '#00f3ff'; // Cyan
            else ctx.fillStyle = '#0F0'; // Classic Matrix Green

            ctx.fillText(text, i * fontSize, rainDrops[i] * fontSize);

            if (rainDrops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                rainDrops[i] = 0;
            }
            rainDrops[i]++;
        }
    };

    setInterval(draw, 30);

    window.addEventListener('resize', () => {
        cw = window.innerWidth;
        ch = window.innerHeight;
        canvas.width = cw;
        canvas.height = ch;
        columns = cw / fontSize;
        rainDrops = [];
        for (let x = 0; x < columns; x++) {
            rainDrops[x] = 1;
        }
    });
});
