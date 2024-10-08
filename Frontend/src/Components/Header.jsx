import React from 'react'
import { Link } from 'react-router-dom';
import './Header.css'
const Header = () => {
  return (
    <>
        <nav className='navbar'>
    
            <img  className='logo'
            src="/logo.png" // Replace with actual image URL
            alt="logo"
          />
      
        <ul className='navbar-menu'>
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/about">About</Link>
        </li>
        <li>
          <Link to="/detection">Detect</Link>
        </li>
        <li>
          <Link to="/how-to-use">How To Use</Link>
        </li>
        </ul>
          
        </nav>
    </>
  )
}

export default Header;