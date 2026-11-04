import { useCallback } from 'react';
import { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import HowItWorks from './pages/HowItWorks';
import About from './pages/About';
import Dashboard from './pages/Dashboard';
import { useBackendHealth } from './hooks/useBackendHealth';

export type Route = 'home' | 'analyze' | 'how' | 'about' | 'dashboard';

export default function App() {
  const [route, setRoute] = useState<Route>('home');
  const { health, loading, error, refresh } = useBackendHealth();

  const navigate = useCallback(
    (to: Route) => {
      setRoute(to);
      window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
    },
    [setRoute],
  );

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar route={route} onNavigate={navigate} health={health ?? null} healthLoading={loading} />
      <main className="flex-1">
        {route === 'home' && (
          <Home onNavigate={navigate} />
        )}
        {route === 'analyze' && (
          <Analyze health={health ?? null} healthLoading={loading} healthError={error} onRefreshHealth={refresh} />
        )}
        {route === 'how' && <HowItWorks onNavigate={navigate} />}
        {route === 'about' && <About onNavigate={navigate} />}
        {route === 'dashboard' && (
          <Dashboard health={health ?? null} healthLoading={loading} healthError={error} />
        )}
      </main>
      <Footer onNavigate={navigate} />
    </div>
  );
}
