const savedTheme = localStorage.getItem('theme');
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

const themeToggle = document.getElementById('themeToggle');
function applyTheme() {
    const isLight = document.documentElement.dataset.theme === 'light';
    themeToggle.textContent = isLight ? '\uD83C\uDF19' : '\u2600\uFE0F';
}
themeToggle.addEventListener('click', () => {
    const isLight = document.documentElement.dataset.theme === 'light';
    document.documentElement.dataset.theme = isLight ? 'dark' : 'light';
    localStorage.setItem('theme', isLight ? 'dark' : 'light');
    applyTheme();
});
applyTheme();

async function predictSample(label) {
    const url = '/static/samples/' + label + '.jpg';
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error('Sample image not found: ' + label + '.jpg');
        const blob = await res.blob();
        const file = new File([blob], label + '.jpg', { type: 'image/jpeg' });
        handleFile(file);
        predictBtn.disabled = false;
        predictBtn.click();
    } catch (err) {
        result.hidden = false;
        resultText.textContent = 'Error: ' + err.message;
        resultText.className = 'tumor';
        probBox.innerHTML = '';
    }
}

const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const result = document.getElementById('result');
const resultText = document.getElementById('resultText');
const probBox = document.getElementById('probBox');
const heatmapBox = document.getElementById('heatmapBox');
const heatmapImage = document.getElementById('heatmapImage');
const heatmapOriginal = document.getElementById('heatmapOriginal');

let selectedFile = null;

uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#4fc3f7';
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'rgba(79, 195, 247, 0.6)';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    heatmapBox.hidden = true;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.hidden = false;
        uploadPrompt.style.display = 'none';
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';
    result.hidden = false;
    resultText.textContent = 'Please wait...';
    resultText.className = '';
    probBox.innerHTML = '';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        const isTumor = data.class_name !== 'notumor';
        resultText.textContent = data.class_name + '  (' + data.confidence + '%)';
        resultText.className = isTumor ? 'tumor' : 'no-tumor';

        probBox.innerHTML = '<h3>Confidence per class</h3>';
        const grid = document.createElement('div');
        grid.className = 'prob-grid';
        for (const [label, conf] of Object.entries(data.probabilities)) {
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML =
                '<span class="prob-label">' + label + '</span>' +
                '<span class="prob-val">' + conf + '%</span>' +
                '<div class="bar"><div style="width:' + conf + '%"></div></div>';
            grid.appendChild(item);
        }
        probBox.appendChild(grid);

        if (data.heatmap) {
            heatmapImage.src = 'data:image/png;base64,' + data.heatmap;
            if (preview.src) heatmapOriginal.src = preview.src;
            heatmapBox.hidden = false;
        } else {
            heatmapBox.hidden = true;
        }
    } catch (err) {
        resultText.textContent = 'Error: ' + err.message;
        resultText.className = 'tumor';
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
});