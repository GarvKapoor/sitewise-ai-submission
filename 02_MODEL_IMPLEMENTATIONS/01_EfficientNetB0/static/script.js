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

analyzeButton.addEventListener("click", async () => {

    if (!selectedFile) {
        return;
    }


    analyzeButton.disabled = true;

    results.classList.add("hidden");

    processing.classList.remove("hidden");

    processingText.textContent =
        "INITIALIZING AI PIPELINE...";


    await delay(500);

    processingText.textContent =
        "RUNNING EFFICIENTNETB0 LAND CLASSIFICATION...";


    await delay(700);

    processingText.textContent =
        "EXTRACTING VISUAL SITE FEATURES...";


    await delay(600);

    processingText.textContent =
        "GENERATING SITE RECOMMENDATIONS...";


    const formData = new FormData();

    formData.append("image", selectedFile);


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

        await delay(500);

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

});


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