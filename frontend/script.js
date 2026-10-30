// Render-ல் உள்ள உங்கள் Backend URL-ஐ இங்கு உள்ளிடவும்
const API_URL = "https://sentiment-analysis-api-heky.onrender.com/analyze";

// 1. Auth & Session Management (Local Persistence)
window.addEventListener("DOMContentLoaded", () => {
  checkUserSession();
});

function checkUserSession() {
  const user = localStorage.getItem("sentix_user");
  const authBtn = document.getElementById("authBtn");
  const userBadge = document.getElementById("userBadge");
  const userNameDisplay = document.getElementById("userNameDisplay");

  if (user) {
    authBtn.classList.add("hidden");
    userBadge.classList.remove("hidden");
    userNameDisplay.innerText = user.split("@")[0];
  } else {
    authBtn.classList.remove("hidden");
    userBadge.classList.add("hidden");
  }
}

function openModal() {
  document.getElementById("loginModal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("loginModal").classList.add("hidden");
}

function handleAuth(e) {
  e.preventDefault();
  const email = document.getElementById("authEmail").value.trim();
  if (email) {
    localStorage.setItem("sentix_user", email);
    closeModal();
    checkUserSession();
  }
}

function logout() {
  localStorage.removeItem("sentix_user");
  checkUserSession();
}

// 2. Character Counter & Sample Templates
function updateCharCount() {
  const len = document.getElementById("reviewInput").value.length;
  document.getElementById("charCount").innerText = `${len} characters`;
}

function fillSample(type) {
  const input = document.getElementById("reviewInput");
  if (type === 1) {
    input.value = "The display is incredibly sharp and battery easily lasts 2 full days. Exceptional build quality and worth every rupee!";
  } else if (type === 2) {
    input.value = "Package was severely delayed by 10 days, outer box was completely crushed and the item had scratches.";
  } else if (type === 3) {
    input.value = "Customer service was very polite, replaced my damaged item within 24 hours without any hassle.";
  }
  updateCharCount();
}

// 3. Main NLP Predict Request
async function analyzeReview() {
  const text = document.getElementById("reviewInput").value.trim();
  if (!text) {
    alert("தயவுசெய்து முதலில் ஒரு விமர்சனத்தை உள்ளிடவும்!");
    return;
  }

  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");
  const resultBox = document.getElementById("resultBox");

  // Loading State
  btnText.classList.add("hidden");
  btnSpinner.classList.remove("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      throw new Error("Backend response error");
    }

    const data = await response.json();

    // Show Results
    resultBox.classList.remove("hidden");

    // Sentiment Tag
    const pill = document.getElementById("sentimentPill");
    pill.innerText = data.sentiment;
    pill.className = `sentiment-pill ${data.sentiment.toLowerCase()}`;

    // Metric Values
    document.getElementById("sentimentVal").innerText = data.sentiment;
    document.getElementById("confidenceVal").innerText = data.confidence;
    
    // Confidence Bar animation
    const confFloat = parseFloat(data.confidence);
    document.getElementById("confidenceBar").style.width = `${confFloat}%`;

    // Keywords Tags
    const keywordsContainer = document.getElementById("keywordsList");
    keywordsContainer.innerHTML = "";
    data.topic_keywords.forEach(word => {
      const tag = document.createElement("span");
      tag.className = "keyword-tag";
      tag.innerHTML = `<i class="fa-solid fa-hashtag"></i> ${word}`;
      keywordsContainer.appendChild(tag);
    });

  } catch (error) {
    alert("API-ஐ இணைக்க முடியவில்லை. Render சர்வர் இயங்குகிறதா என்பதை உறுதிப்படுத்தவும் (Cold start இருப்பின் 30 வினாடிகளில் மீண்டும் முயலவும்).");
    console.error(error);
  } finally {
    btnText.classList.remove("hidden");
    btnSpinner.classList.add("hidden");
  }
}