/**
 * TTS DooMax — bouton flottant qui lit la dernière réponse.
 */
(function () {
  let voixFR = null;
  let enLecture = false;

  function chargerVoix() {
    const voix = speechSynthesis.getVoices();
    voixFR =
      voix.find((v) => v.lang === "fr-FR" && v.name.includes("Google")) ||
      voix.find((v) => v.lang === "fr-FR") ||
      voix.find((v) => v.lang.startsWith("fr")) ||
      voix[0];
  }
  speechSynthesis.onvoiceschanged = chargerVoix;
  chargerVoix();

  // Bouton flottant en bas à gauche
  const btn = document.createElement("button");
  btn.id = "tts-doomax";
  btn.textContent = "🔊";
  btn.title = "DooMax lit la dernière réponse";
  btn.style.cssText =
    "position:fixed;bottom:20px;left:20px;z-index:99999;" +
    "width:48px;height:48px;border-radius:50%;" +
    "background:#1D9E75;border:2px solid #fff;color:#fff;" +
    "font-size:22px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,0.3);" +
    "display:flex;align-items:center;justify-content:center;" +
    "transition:transform 0.2s;";
  btn.onmouseenter = () => (btn.style.transform = "scale(1.15)");
  btn.onmouseleave = () => (btn.style.transform = "scale(1)");

  btn.onclick = () => {
    if (enLecture) {
      speechSynthesis.cancel();
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
      return;
    }

    // Trouver la dernière réponse visible
    const messages = document.querySelectorAll(
      "[class*='message'] p, [class*='step'] p, [class*='content'] p, .markdown-body p, .prose p"
    );
    if (!messages.length) return;

    // Prendre les 10 derniers paragraphes (la dernière réponse)
    const derniers = Array.from(messages).slice(-10);
    const texte = derniers.map((p) => p.textContent).join(". ").replace(/\s+/g, " ").trim();
    if (!texte) return;

    const utterance = new SpeechSynthesisUtterance(texte);
    utterance.voice = voixFR;
    utterance.lang = "fr-FR";
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      btn.textContent = "⏹️";
      btn.style.background = "#dc2626";
      enLecture = true;
    };
    utterance.onend = () => {
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
    };
    utterance.onerror = () => {
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
    };

    speechSynthesis.speak(utterance);
  };

  document.body.appendChild(btn);
})();
