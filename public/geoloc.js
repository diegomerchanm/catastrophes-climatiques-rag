// SAEARCH — Geolocalisation utilisateur
// Envoie la position GPS au backend via un message cache au demarrage

if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            // Stocker dans sessionStorage pour utilisation ulterieure
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
