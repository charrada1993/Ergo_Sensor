# Guide de Déploiement : Hébergement de Ergo Sensor

Ce document explique les étapes pour héberger l'application complète Ergo Sensor (Backend IA + Frontend Dashboard + Socket.IO) en production.

> ⚠️ **Pourquoi pas Vercel ou Netlify ?**
> Vercel et Netlify sont conçus pour des architectures *Serverless* (fonctions qui s'éteignent après quelques secondes). Or, Ergo Sensor nécessite une connexion continue via **WebSockets** (Socket.IO) pour envoyer le flux de données en direct, ainsi que la mémoire nécessaire pour charger les modèles d'Intelligence Artificielle (LightGBM). Le Serverless n'est pas du tout adapté pour cela car il coupe les connexions WebSocket et manque de RAM.

> ✅ **La Solution : Render**
> **Render.com** est la plateforme idéale pour ce projet. Elle supporte nativement Python, les WebSockets continus, et installe automatiquement les dépendances (`requirements.txt`).

---

## Étapes de Déploiement sur Render.com

### Étape 1 : Préparation du code (Déjà fait !)
L'application est prête. Assurez-vous d'avoir poussé tout votre code sur GitHub, avec les fichiers indispensables à la racine :
- `requirements.txt` (Contient Flask, Flask-SocketIO, lightgbm, etc.)
- `app.py` (Le serveur principal)
- `models/` (Le dossier contenant vos modèles IA entraînés)

### Étape 2 : Créer le Web Service
1. Allez sur [Render.com](https://render.com/) et créez un compte.
2. Cliquez sur **New +** en haut à droite, puis sélectionnez **Web Service**.
3. Connectez votre compte GitHub et sélectionnez le dépôt `Ergo_Sensor`.

### Étape 3 : Configuration du Web Service
Remplissez les champs de configuration avec ces valeurs :
- **Name** : `ergo-sensor` (ou le nom de votre choix)
- **Region** : Choisissez la région la plus proche de vous (ex: Frankfurt).
- **Environment** : `Python 3`
- **Build Command** : `pip install -r requirements.txt`
- **Start Command** : `gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app`

### Étape 4 : Variables d'Environnement (CRUCIAL)
Toujours sur la page de configuration de Render, descendez jusqu'à la section **Environment Variables** et ajoutez :
- `PYTHON_VERSION` : `3.11.0` (C'est obligatoire car les bibliothèques WebSockets ne sont pas encore compatibles avec Python 3.14).

### Étape 5 : Déployer
1. Choisissez un plan tarifaire. (Le plan *Starter* avec plus de RAM est fortement recommandé pour charger les modèles IA).
2. Cliquez sur **Create Web Service**.

Render va télécharger le dépôt GitHub, installer les dépendances et lancer l'application. Une fois terminé, vous obtiendrez une URL publique (ex: `https://ergo-sensor.onrender.com`) que vous pourrez partager ou utiliser pour connecter vos capteurs ESP32 !

---

### Configuration finale des capteurs
Une fois l'URL Render générée, vous devrez simplement mettre à jour le code de vos capteurs (ou votre testeur de charge) pour pointer vers l'URL en production plutôt que `localhost` :

*Avant :* `http://127.0.0.1:5000/api/data`
*Après :* `https://ergo-sensor.onrender.com/api/data`
