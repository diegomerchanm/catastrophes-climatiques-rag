// SAEARCH — Geolocalisation utilisateur
if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
        function(position) {
            var lat = position.coords.latitude;
            var lon = position.coords.longitude;
            sessionStorage.setItem('saearch_lat', lat);
            sessionStorage.setItem('saearch_lon', lon);
            console.log('SAEARCH geoloc:', lat, lon);
        },
        function(error) {
            console.log('SAEARCH geoloc refused:', error.message);
        },
        { enableHighAccuracy: false, timeout: 5000 }
    );
}

// SAEARCH — TTS DooMax (bouton flottant 🔊)
(function () {
  var voixFR = null;
  var enLecture = false;

  function chargerVoix() {
    var voix = speechSynthesis.getVoices();
    voixFR =
      voix.find(function(v) { return v.lang === "fr-FR"; }) ||
      voix.find(function(v) { return v.lang.startsWith("fr"); }) ||
      voix[0];
  }
  if (typeof speechSynthesis !== 'undefined') {
    speechSynthesis.onvoiceschanged = chargerVoix;
    chargerVoix();
  }

  var btn = document.createElement("button");
  btn.textContent = "🔊";
  btn.title = "DooMax lit la dernière réponse";
  btn.style.cssText =
    "position:fixed;bottom:20px;left:20px;z-index:99999;" +
    "width:48px;height:48px;border-radius:50%;" +
    "background:#1D9E75;border:2px solid #fff;color:#fff;" +
    "font-size:22px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,0.3);" +
    "display:flex;align-items:center;justify-content:center;";

  btn.onclick = function() {
    // Toujours arrêter d'abord (même si enLecture a raté)
    if (speechSynthesis.speaking || enLecture) {
      speechSynthesis.cancel();
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
      return;
    }

    // Si du texte est sélectionné à la souris → lire la sélection
    var selection = window.getSelection().toString().trim();
    if (selection && selection.length > 20) {
      var texte = selection.substring(0, 3000);
    } else {
    // Sinon : extraire la dernière réponse de DooMax
    var tout = document.body.innerText || "";
    var lignes = tout.split("\n");

    // Trouver la DERNIÈRE occurrence d'un badge de route (= début dernière réponse)
    var debut = 0;
    for (var i = lignes.length - 1; i >= 0; i--) {
      var l = lignes[i].trim();
      if (l.indexOf("Agent --") !== -1 || l.indexOf("RAG --") !== -1 || l.indexOf("Chat --") !== -1) {
        debut = i + 1;
        break;
      }
    }

    // Filtrer les lignes après le badge : garder que le vrai texte
    var JUNK = ["RAG", "Meteo", "Calcul", "Scoring", "Email", "Agent", "Chat", "Web",
                "ML", "5%", "60%", "23%", "outils", "Tokens", "Source:", "Page:",
                ".pdf", "v1.0", "v2.0", "$0.", "in:", "out:", "Outils externes"];
    var texte = "";
    for (var i = debut; i < lignes.length; i++) {
      var l = lignes[i].trim();
      if (l.length < 15) continue;
      var junk = false;
      for (var j = 0; j < JUNK.length; j++) {
        if (l.indexOf(JUNK[j]) !== -1) { junk = true; break; }
      }
      if (!junk) texte += l + ". ";
    }
    texte = texte.substring(0, 3000).trim();
    } // fin du else (pas de sélection)
    if (!texte) return;

    // Détecter la langue dominante par comptage de mots
    var mots = texte.toLowerCase().split(/\s+/);
    var frWords = ["les","des","dans","pour","avec","sur","une","par","sont","cette","qui","est","aux","ses","ont","mais","que","plus"];
    var enWords = ["the","is","are","was","were","with","from","this","that","have","has","been","will","would","could","their","which","also"];
    var esWords = ["las","los","por","con","una","del","para","como","sobre","entre","pero","desde","tiene","puede","esta","estos"];
    var deWords = ["und","die","das","ist","mit","für","ein","von","den","dem","sich","auf","auch","als","nach","wird"];
    var countFR = 0, countEN = 0, countES = 0, countDE = 0;
    for (var w = 0; w < mots.length; w++) {
      if (frWords.indexOf(mots[w]) !== -1) countFR++;
      if (enWords.indexOf(mots[w]) !== -1) countEN++;
      if (esWords.indexOf(mots[w]) !== -1) countES++;
      if (deWords.indexOf(mots[w]) !== -1) countDE++;
    }
    var lang = "fr-FR";
    var maxCount = countFR;
    if (countEN > maxCount) { lang = "en-US"; maxCount = countEN; }
    if (countES > maxCount) { lang = "es-ES"; maxCount = countES; }
    if (countDE > maxCount) { lang = "de-DE"; maxCount = countDE; }

    var voix = speechSynthesis.getVoices();
    var voixChoisie = voix.find(function(v) { return v.lang === lang; }) ||
                      voix.find(function(v) { return v.lang.startsWith(lang.split("-")[0]); }) ||
                      voixFR;

    var utterance = new SpeechSynthesisUtterance(texte);
    utterance.voice = voixChoisie;
    utterance.lang = lang;
    utterance.rate = 1.0;

    utterance.onstart = function() {
      btn.textContent = "⏹️";
      btn.style.background = "#dc2626";
      enLecture = true;
    };
    utterance.onend = function() {
      speechSynthesis.cancel();
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
    };
    utterance.onerror = function() {
      speechSynthesis.cancel();
      btn.textContent = "🔊";
      btn.style.background = "#1D9E75";
      enLecture = false;
    };

    speechSynthesis.speak(utterance);
  };

  document.body.appendChild(btn);
  console.log('SAEARCH TTS: bouton 🔊 ajouté');
})();
