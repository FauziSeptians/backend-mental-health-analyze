# Mental Health App

A comprehensive mental health application designed to support users in their mental wellness journey.

## Overview

This Mental Health App provides a safe, accessible platform for individuals to monitor, understand, and improve their mental health. Our goal is to break down barriers to mental health care by offering valuable resources, mood tracking, guided exercises.

## Features

### 🧠 HomePage Information
- Showing information about mental health issues
- Showing information about what we do in this website

### 📚 Analyzing user mental health using statement analysis
- Single analyze text
- Bulk analyze text
- Statement analysis using AI mental bert model (Deep learning)
- Statement analysis using twitter data
- Summary mental health analysis

## Tech Stack

- **Frontend**: Next.js with React
- **Backend**: Flask API routes
- **Model**: Mental Bert
- **Deployment**: Vercel

### Instalasi dan Menjalankan Backend
1. Unduh repositori untuk layanan backend analisis kesehatan mental:
   ```
   git clone https://github.com/FauziSeptians/backend-mental-health-analyze.git
   ```

2. Pasang semua pustaka yang tercantum dalam berkas persyaratan:
   ```
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi backend menggunakan Flask:
   ```
   python app/main.py
   ```

4. Catat alamat layanan backend (http://IP:4000) yang akan digunakan dalam pengaturan frontend.
