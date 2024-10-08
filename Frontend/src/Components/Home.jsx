import React from 'react';
import { Link } from 'react-router-dom'; // Assuming you have react-router for navigation
import './HomePage.css'; // Import the CSS for styling

const HomePage = () => {
  return (
    <div className='homepage-container '>

      <main className="hero-section">
        <div className="hero-content">
          <h2>Sign Language Detection</h2>
          <p>
           "Developing a machine learning-based system for real-time sign language detection and translation."
          </p>
          <Link to="/detection">
            <button className="start-button">Start Detection</button>
          </Link>
        </div>
        <div className="hero-image">
          <img
            src="https://i0.wp.com/miro.medium.com/v2/resize:fit:738/1*XrbqBLMR1W3N8mIQCPzPbw.png?ssl=1" // Replace with actual image URL
            alt="Sign Language Detection"
          />
        </div>
      </main>

      <footer className="footer">
        <p>&copy; 2024 Signify | Empowering Communication</p>
      </footer>
    </div>
  );
};

export default HomePage;
