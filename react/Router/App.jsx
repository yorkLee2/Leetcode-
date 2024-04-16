import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import Btn from './Btn';
import './App.css';
import ModeToggler from './ModeToggler';

import Homepage from "./Homepage";
import AboutMe from "./AboutMe";
import { Routes, Route, Link } from 'react-router-dom';

function App() {
  return (
      <div className="App">
        <nav className="nav">
          <Link to="/" className="nav-item">Homepage</Link>
          <Link to="/about-me" className="nav-item">About Me</Link>
        </nav>
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/about-me" element={<AboutMe />} />
        </Routes>
      </div>
         
  );
}

export default App;
