import React from 'react';
import Header from './Header/Header';
import ImageUpload from './ImageUpload';
import Footer from './Footer/Footer';
import './App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <main>
        <ImageUpload />
      </main>
      <Footer />
    </div>
  );
}

export default App;



