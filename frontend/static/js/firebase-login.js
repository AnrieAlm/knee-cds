'use strict'

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.9.0/firebase-app.js"
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut } from "https://www.gstatic.com/firebasejs/12.9.0/firebase-auth.js"

import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.9.0/firebase-auth.js"

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

    document.getElementById("sign-up").addEventListener('click', function () {
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

    document.getElementById("login").addEventListener('click', function () {
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
    document.getElementById("sign-out").addEventListener('click', function () {
    signOut(auth).then(() => {
        document.cookie = "token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
        window.location = "/login";
    });
});

onAuthStateChanged(auth, (user) => {
    document.getElementById("sign-out").hidden = !user;
});
});