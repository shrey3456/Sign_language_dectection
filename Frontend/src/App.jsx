import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './Components/Header';
import Home from './Components/Home';
import About from './Components/About';
import Detection from './Components/Detection';
import HowToUse from './Components/HowToUse';

function App() {
  return (
    <Router>
      <div>
        <Header />
        <div className="sections">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/how-to-use" element={<HowToUse />} />
            <Route path="/detection" element={<Detection />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
