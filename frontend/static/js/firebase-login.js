'use strict'

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.9.0/firebase-app.js"
import {
    getAuth,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut
} from "https://www.gstatic.com/firebasejs/12.9.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyAG87QkP2r9N8jXl8gM0fCD001mwXfaQdA",
    authDomain: "anriel-01.firebaseapp.com",
    projectId: "anriel-01",
    storageBucket: "anriel-01.firebasestorage.app",
    messagingSenderId: "23652264454",
    appId: "1:23652264454:web:6fef33746210245d25f187"
};

window.addEventListener("load", function () {
    const app = initializeApp(firebaseConfig)
    const auth = getAuth()

    // booklet pattern: read the cookie on load to set UI visibility
    updateUI(document.cookie)

    const signUpBtn = document.getElementById("sign-up");
    if (signUpBtn) {
        signUpBtn.addEventListener('click', function () {
            const email = document.getElementById("email").value
            const password = document.getElementById("password").value
            createUserWithEmailAndPassword(auth, email, password)
                .then((userCredential) => {
                    userCredential.user.getIdToken().then((token) => {
                        document.cookie = "token=" + token + ";path=/;SameSite=Strict";
                        window.location = "/";
                    });
                })
                .catch((error) => {
                    const errBox = document.getElementById("login-error");
                    if (errBox) errBox.textContent = error.message;
                })
        })
    }

    const loginBtn = document.getElementById("login");
    if (loginBtn) {
        loginBtn.addEventListener('click', function () {
            const email = document.getElementById("email").value
            const password = document.getElementById("password").value
            signInWithEmailAndPassword(auth, email, password)
                .then((userCredential) => {
                    userCredential.user.getIdToken().then((token) => {
                        document.cookie = "token=" + token + ";path=/;SameSite=Strict";
                        window.location = "/";
                    });
                })
                .catch((error) => {
                    const errBox = document.getElementById("login-error");
                    if (errBox) errBox.textContent = error.message;
                });
        });
    }

    const signOutBtn = document.getElementById("sign-out");
    if (signOutBtn) {
        signOutBtn.addEventListener('click', function () {
            signOut(auth).then(() => {
                document.cookie = "token=;path=/;SameSite=Strict";
                window.location = "/login";
            });
        });
    }
});

// booklet helper: show/hide sign-out button based on the cookie token
function updateUI(cookie) {
    var token = parseCookieToken(cookie);
    var signOutBtn = document.getElementById("sign-out");
    if (token.length > 0) {
        if (signOutBtn) signOutBtn.hidden = false;
    } else {
        if (signOutBtn) signOutBtn.hidden = true;
    }
}

// booklet helper: extract the token value from the cookie string
function parseCookieToken(cookie) {
    var strings = cookie.split(';');
    for (let i = 0; i < strings.length; i++) {
        var temp = strings[i].split('=');
        if (temp[0].trim() == "token")
            return temp[1];
    }
    return ""
}