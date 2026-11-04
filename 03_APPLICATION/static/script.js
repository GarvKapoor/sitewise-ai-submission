const imageInput = document.getElementById("imageInput");
const selectButton = document.getElementById("selectButton");
const analyzeButton = document.getElementById("analyzeButton");

const dropZone = document.getElementById("dropZone");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");

const previewImage = document.getElementById("previewImage");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");

const processing = document.getElementById("processing");
const processingText = document.getElementById("processingText");

const results = document.getElementById("results");

let selectedFile = null;
let currentModel = "efficientnet"; // "efficientnet" | "unet" | "papnet"

/* =========================
   MODEL SELECTION METADATA
========================= */

const modelMeta = {
    efficientnet: {
        name: "EfficientNetB0",
        cardLabel: "MODEL 1 / LAND CLASSIFICATION (EFFICIENTNET)",
        distLabel: "MODEL 1 — EFFICIENTNET-B0",
        stepTitle: "LAND CLASSIFICATION",
        stepDesc: "EfficientNetB0 analyzes the image and predicts the land category with a confidence score across ten classes.",
        explanation: "The first model performs land classification using an EfficientNetB0 architecture. Its output contains both the predicted land type and probability distribution across the supported land classes.",
        loadingText: "RUNNING EFFICIENTNETB0 LAND CLASSIFICATION..."
    },
    unet: {
        name: "U-Net",
        cardLabel: "MODEL 1 / LAND COVER SEGMENTATION (U-NET)",
        distLabel: "MODEL 1 — U-NET SEGMENTATION",
        stepTitle: "LAND SEGMENTATION (U-NET)",
        stepDesc: "U-Net convolutional encoder-decoder with skip connections extracts multi-scale spatial terrain features and generates pixel-wise land cover segmentations.",
        explanation: "The first model performs land cover analysis using a deep U-Net encoder-decoder architecture with skip connections preserving spatial detail to map pixel-wise terrain characteristics.",
        loadingText: "RUNNING U-NET TERRAIN SEGMENTATION..."
    },
    papnet: {
        name: "PAPNet",
        cardLabel: "MODEL 1 / LAND CLASSIFICATION (PAPNET)",
        distLabel: "MODEL 1 — PAPNET (PATTERN-AFFINITIVE PROPAGATION)",
        stepTitle: "LAND CLASSIFICATION (PAPNET)",
        stepDesc: "PAPNet (Pattern-Affinitive Propagation) analyzes satellite image patches using attention-guided affinity modules to classify land cover across ten classes.",
        explanation: "The first model performs land classification using a Pattern-Affinitive Propagation Network (PAPNet). It computes non-local spatial pattern affinity matrices across satellite image patches to classify land use across ten EuroSAT classes.",
        loadingText: "RUNNING PAPNET LAND CLASSIFICATION..."
    }
};

function setModel(modelKey) {
    if (!modelMeta[modelKey]) return;
    currentModel = modelKey;

    document.querySelectorAll(".nav-model-btn").forEach(btn => {
        if (btn.dataset.model === modelKey) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    const meta = modelMeta[modelKey];
    const lbl = document.getElementById("model1CardLabel");
    if (lbl) lbl.textContent = meta.cardLabel;

    const dlbl = document.getElementById("model1DistLabel");
    if (dlbl) dlbl.textContent = meta.distLabel;

    const st = document.getElementById("step2Title");
    if (st) st.textContent = meta.stepTitle;

    const sd = document.getElementById("step2Desc");
    if (sd) sd.textContent = meta.stepDesc;

    const exp = document.getElementById("explanationModel1");
    if (exp) exp.textContent = meta.explanation;

    // If an image was already analyzed and results are visible, re-analyze with the new model
    if (selectedFile && !results.classList.contains("hidden")) {
        analyzeImage();
    }
}

document.querySelectorAll(".nav-model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        setModel(btn.dataset.model);
    });
});


/* =========================
   FILE SELECTION
========================= */

selectButton.addEventListener("click", () => {
    imageInput.click();
});


imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {
        handleFile(file);
    }

});


/* =========================
   DRAG & DROP
========================= */

dropZone.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropZone.classList.add("dragging");

});


dropZone.addEventListener("dragleave", () => {

    dropZone.classList.remove("dragging");

});


dropZone.addEventListener("drop", (event) => {

    event.preventDefault();

    dropZone.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }

});


dropZone.addEventListener("click", () => {

    if (!selectedFile) {
        imageInput.click();
    }

});


/* =========================
   HANDLE IMAGE
========================= */

function handleFile(file) {

    const validTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!validTypes.includes(file.type)) {

        alert("Please upload a JPG, JPEG or PNG image.");

        return;
    }


    selectedFile = file;

    fileName.textContent = file.name;

    const reader = new FileReader();


    reader.onload = function(event) {

        previewImage.src = event.target.result;

        previewImage.classList.add("visible");

        uploadPlaceholder.style.display = "none";

        fileInfo.classList.add("visible");

        analyzeButton.disabled = false;

    };


    reader.readAsDataURL(file);

}


/* =========================
   ANALYZE
========================= */

async function analyzeImage() {

    if (!selectedFile) {
        return;
    }

    analyzeButton.disabled = true;

    results.classList.add("hidden");

    processing.classList.remove("hidden");

    processingText.textContent =
        "INITIALIZING AI PIPELINE...";

    await delay(350);

    const activeMeta = modelMeta[currentModel] || modelMeta.efficientnet;
    processingText.textContent = activeMeta.loadingText;

    await delay(500);

    processingText.textContent =
        "EXTRACTING VISUAL SITE FEATURES...";

    await delay(400);

    processingText.textContent =
        "GENERATING SITE RECOMMENDATIONS...";

    const formData = new FormData();

    formData.append("image", selectedFile);
    formData.append("model", currentModel);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || "Site analysis failed."
            );

        }

        processingText.textContent =
            "ANALYSIS COMPLETE.";

        await delay(350);

        processing.classList.add("hidden");

        renderResults(data);

        results.classList.remove("hidden");

        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {

        processing.classList.add("hidden");

        alert(error.message);

    } finally {

        analyzeButton.disabled = false;

    }

}

analyzeButton.addEventListener("click", analyzeImage);


/* =========================
   RENDER RESULTS
========================= */

function renderResults(data) {

    /*
        LAND TYPE
    */

    document.getElementById("landType").textContent =
        formatText(data.land_type);


    const confidence =
        Number(data.land_confidence || 0) * 100;


    document.getElementById("landConfidence").textContent =
        `${confidence.toFixed(2)}%`;


    document.getElementById("confidenceGauge").textContent =
        `${confidence.toFixed(1)}%`;


    animateGauge(
        "confidenceCircle",
        confidence
    );


    /*
        BEST RECOMMENDATION
    */

    const recommendation =
        data.recommendation ||
        getBestRecommendation(data);


    document.getElementById("recommendation").textContent =
        formatText(recommendation);


    const suitability =
        Number(data.suitability || 0) * 100;


    document.getElementById("suitabilityValue").textContent =
        `${suitability.toFixed(2)}%`;


    setTimeout(() => {

        document.getElementById("suitabilityBar").style.width =
            `${Math.min(suitability, 100)}%`;

    }, 100);


    const decision =
        data.decision ||
        getDecision(suitability);


    document.getElementById("decision").textContent =
        decision;


    /*
        RECOMMENDATION RANKING
    */

    renderRecommendations(
        data.recommendations ||
        data.recommendation_scores ||
        {}
    );


    /*
        LAND PROBABILITIES
    */

    renderLandProbabilities(
        data.land_probabilities ||
        {}
    );


    /*
        VISUAL FEATURES
    */

    renderFeatures(
        data.site_features ||
        {}
    );


    /*
        MODEL METADATA & U-NET SEGMENTATION DISPLAY
    */

    const usedModel = data.model_used || currentModel;
    const meta = modelMeta[usedModel] || modelMeta.efficientnet;
    const cardLbl = document.getElementById("model1CardLabel");
    if (cardLbl) cardLbl.textContent = meta.cardLabel;

    const distLbl = document.getElementById("model1DistLabel");
    if (distLbl) distLbl.textContent = meta.distLabel;

    const unetSection = document.getElementById("unetSegSection");
    const unetImg = document.getElementById("unetMaskImg");
    if (unetSection && unetImg) {
        if (data.segmentation_overlay && usedModel === "unet") {
            unetImg.src = data.segmentation_overlay;
            unetSection.classList.remove("hidden");
        } else {
            unetSection.classList.add("hidden");
        }
    }

}


/* =========================
   RECOMMENDATIONS
========================= */

function renderRecommendations(scores) {

    const container =
        document.getElementById("recommendationList");

    container.innerHTML = "";


    let entries = [];


    if (Array.isArray(scores)) {

        entries = scores.map((item) => {

            return [
                item.name ||
                item.recommendation ||
                item.label,

                Number(
                    item.score ||
                    item.suitability ||
                    item.value ||
                    0
                )
            ];

        });

    } else {

        entries = Object.entries(scores);

    }


    entries.sort((a, b) => b[1] - a[1]);


    entries.forEach((entry, index) => {

        const name = entry[0];

        let score = Number(entry[1]);

        if (score <= 1) {
            score *= 100;
        }


        const item =
            document.createElement("div");

        item.className =
            "recommendation-item";


        item.innerHTML = `

            <div class="rank-number">
                ${String(index + 1).padStart(2, "0")}
            </div>

            <div>

                <div class="rank-name">
                    ${formatText(name)}
                </div>

                <div class="rank-bar">
                    <span></span>
                </div>

            </div>

            <div class="rank-score">
                ${score.toFixed(2)}%
            </div>

        `;


        container.appendChild(item);


        setTimeout(() => {

            const bar =
                item.querySelector(".rank-bar span");

            bar.style.width =
                `${Math.min(score, 100)}%`;

        }, 100 + index * 100);

    });

}


/* =========================
   LAND PROBABILITIES
========================= */

function renderLandProbabilities(probabilities) {

    const container =
        document.getElementById("landProbabilities");

    container.innerHTML = "";


    let entries =
        Object.entries(probabilities);


    entries.sort((a, b) => b[1] - a[1]);


    entries.forEach((entry, index) => {

        let score = Number(entry[1]);

        if (score <= 1) {
            score *= 100;
        }


        const item =
            document.createElement("div");

        item.className =
            "probability-item";


        item.innerHTML = `

            <div class="probability-top">

                <span>
                    ${formatText(entry[0])}
                </span>

                <span>
                    ${score.toFixed(2)}%
                </span>

            </div>

            <div class="probability-bar">

                <span></span>

            </div>

        `;


        container.appendChild(item);


        setTimeout(() => {

            item.querySelector(
                ".probability-bar span"
            ).style.width =
                `${Math.min(score, 100)}%`;

        }, 100 + index * 70);

    });

}


/* =========================
   SITE FEATURES
========================= */

function renderFeatures(features) {

    const container =
        document.getElementById("siteFeatures");

    container.innerHTML = "";


    Object.entries(features).forEach(
        ([name, value]) => {

            const item =
                document.createElement("div");

            item.className =
                "feature";


            let numeric =
                Number(value);


            item.innerHTML = `

                <div class="feature-name">
                    ${formatText(name)}
                </div>

                <div class="feature-value">
                    ${numeric.toFixed(4)}
                </div>

            `;


            container.appendChild(item);

        }
    );

}


/* =========================
   HELPERS
========================= */

function formatText(value) {

    if (!value) {
        return "—";
    }


    return String(value)
        .replaceAll("_", " ")
        .replace(/\b\w/g, char =>
            char.toUpperCase()
        );

}


function getBestRecommendation(data) {

    const scores =
        data.recommendations ||
        data.recommendation_scores ||
        {};


    const entries =
        Object.entries(scores);


    if (!entries.length) {
        return "Site Recommendation";
    }


    entries.sort((a, b) => b[1] - a[1]);


    return entries[0][0];

}


function getDecision(score) {

    return score >= 50
        ? "HIGH SUITABILITY"
        : score >= 25
            ? "MODERATE SUITABILITY"
            : "LOW SUITABILITY";

}


function animateGauge(id, percentage) {

    const circle =
        document.getElementById(id);

    const circumference =
        2 * Math.PI * 48;


    const offset =
        circumference -
        (percentage / 100) * circumference;


    circle.style.strokeDasharray =
        circumference;


    circle.style.strokeDashoffset =
        circumference;


    requestAnimationFrame(() => {

        circle.style.strokeDashoffset =
            offset;

    });

}


function delay(ms) {

    return new Promise(
        resolve => setTimeout(resolve, ms)
    );

}