// Render அல்லது Railway-ல் நீங்கள் ஹோஸ்ட் செய்யும் API URL-ஐ இங்கு உள்ளிடவும்
const API_URL = "https://your-backend-service.onrender.com/predict"; 

async function analyzeReview() {
  const text = document.getElementById("reviewInput").value.trim();
  if (!text) {
    alert("Please enter a review first!");
    return;
  }

  const resultBox = document.getElementById("result");
  resultBox.classList.remove("hidden");
  document.getElementById("sentimentText").innerText = "Analyzing...";
  document.getElementById("confidenceText").innerText = "...";
  document.getElementById("keywordsText").innerText = "...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    document.getElementById("sentimentText").innerText = data.sentiment;
    document.getElementById("confidenceText").innerText = data.confidence;
    document.getElementById("keywordsText").innerText = data.topic_keywords.join(", ");
  } catch (error) {
    alert("Error connecting to backend API. Ensure the backend URL is active.");
    console.error(error);
  }
}