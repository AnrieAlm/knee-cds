'use strict'

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.9.0/firebase-app.js"
import { 
    getAuth, 
    createUserWithEmailAndPassword, 
    signInWithEmailAndPassword, 
    signOut, 
    onAuthStateChanged 
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
                    document.getElementById("login-error").textContent = error.message;
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
                    document.getElementById("login-error").textContent = error.message;
                });
        });
    }

    const signOutBtn = document.getElementById("sign-out");
    if (signOutBtn) {
        signOutBtn.addEventListener('click', function () {
            signOut(auth).then(() => {
                document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
                window.location = "/login";
            });
        });
    }

    onAuthStateChanged(auth, (user) => {
        const btn = document.getElementById("sign-out");
        if (btn) btn.hidden = !user;
    });
});