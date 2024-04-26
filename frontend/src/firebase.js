import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getAuth, signInWithCustomToken, sendPasswordResetEmail } from "firebase/auth";
import { getFunctions } from "firebase/functions";
import firebaseConfig from './firebaseConfig';

const app = initializeApp(firebaseConfig);

const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);
const functions = getFunctions(app);

export { auth, db, storage, functions, sendPasswordResetEmail, signInWithCustomToken };
