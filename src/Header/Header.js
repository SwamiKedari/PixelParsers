import React from 'react';
import './Header.css';
import logo from './pixel_parsers.jpeg';

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <img src={logo} alt="Dentex Logo" />
      </div>
      <h1>Pixel_Parsers</h1>
      <nav>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">About</a></li>
          <li><a href="#">Contact</a></li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;