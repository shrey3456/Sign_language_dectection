import React from 'react'
import { Link } from 'react-router-dom';
import './Header.css'
const Header = () => {
  return (
    <>
        <nav className='navbar'>
        <h1 className='logo'>Logo</h1>
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