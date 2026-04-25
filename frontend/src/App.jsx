import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cpu, 
  BarChart3, 
  Settings, 
  Activity, 
  Zap, 
  Info,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import EnvironmentCard from './components/EnvironmentCard';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedEnv, setSelectedEnv] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/landing');
      setData(response.data);
    } catch (error) {
      console.error("Error fetching suite data:", error);
      // Fallback data for preview
      setData({
        suite: "Math Escalation — Multi-Environment Suite",
        theme: "Theme #4: Self-Improvement",
        environments: [
          { name: "Math Escalation", description: "10-tier adaptive math curriculum", tools: ["get_problem", "submit_answer"], prefix: "/", docs: "/docs" },
          { name: "Negotiation Arena", description: "Self-play resource negotiation", tools: ["make_offer", "accept_offer"], prefix: "/negotiate", docs: "/negotiate/docs" },
          { name: "Coding Competition", description: "Evolving coding challenges", tools: ["submit_code", "get_status"], prefix: "/code", docs: "/code/docs" }
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const renderContent = () => {
    if (activeTab === 'monitoring') {
      return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ paddingTop: '20px' }}>
          <h2 style={{ fontSize: '2rem', marginBottom: '32px' }}>System Monitoring</h2>
          <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
            <h3 style={{ marginBottom: '20px', color: 'var(--accent-math)' }}>Resource Usage</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
              <div>
                <p style={{ color: 'var(--text-muted)', marginBottom: '12px' }}>CPU Load</p>
                <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                  <motion.div initial={{ width: 0 }} animate={{ width: '42%' }} style={{ height: '100%', background: 'var(--accent-math)' }} />
                </div>
              </div>
              <div>
                <p style={{ color: 'var(--text-muted)', marginBottom: '12px' }}>Memory Usage</p>
                <div style={{ height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                  <motion.div initial={{ width: 0 }} animate={{ width: '68%' }} style={{ height: '100%', background: 'var(--accent-negotiate)' }} />
                </div>
              </div>
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '32px' }}>
            <h3 style={{ marginBottom: '20px', color: 'var(--accent-code)' }}>Active Instances</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border-glass)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '12px 0' }}>Environment</th>
                  <th style={{ padding: '12px 0' }}>Status</th>
                  <th style={{ padding: '12px 0' }}>Uptime</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '16px 0' }}>Math Escalation</td>
                  <td style={{ padding: '16px 0', color: '#32d74b' }}>Running</td>
                  <td style={{ padding: '16px 0' }}>14h 22m</td>
                </tr>
                <tr>
                  <td style={{ padding: '16px 0' }}>Negotiation Arena</td>
                  <td style={{ padding: '16px 0', color: '#32d74b' }}>Running</td>
                  <td style={{ padding: '16px 0' }}>14h 22m</td>
                </tr>
              </tbody>
            </table>
          </div>
        </motion.div>
      );
    }

    if (activeTab === 'settings') {
      return (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ paddingTop: '20px' }}>
          <h2 style={{ fontSize: '2rem', marginBottom: '32px' }}>Global Settings</h2>
          <div className="glass-panel" style={{ padding: '32px', maxWidth: '600px' }}>
            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', marginBottom: '12px', color: 'var(--text-muted)' }}>Concurrent Environments</label>
              <input type="number" defaultValue={8} style={{ 
                width: '100%', 
                background: 'rgba(255,255,255,0.05)', 
                border: '1px solid var(--border-glass)', 
                color: 'white', 
                padding: '12px', 
                borderRadius: '8px' 
              }} />
            </div>
            <div style={{ marginBottom: '32px' }}>
              <label style={{ display: 'block', marginBottom: '12px', color: 'var(--text-muted)' }}>API Port</label>
              <input type="text" defaultValue="8000" disabled style={{ 
                width: '100%', 
                background: 'rgba(255,255,255,0.02)', 
                border: '1px solid var(--border-glass)', 
                color: '#666', 
                padding: '12px', 
                borderRadius: '8px' 
              }} />
            </div>
            <button className="btn-primary">Save Changes</button>
          </div>
        </motion.div>
      );
    }

    return (
      <>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '48px' }}>
          <div>
            <motion.h1 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              style={{ fontSize: '2.5rem', marginBottom: '8px', fontWeight: '800' }}
            >
              System Overview
            </motion.h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>Manage your adaptive RL environments</p>
          </div>
          <motion.button 
            whileHover={{ rotate: 180 }}
            onClick={fetchData}
            style={{ 
              background: 'rgba(255,255,255,0.05)', 
              border: '1px solid var(--border-glass)', 
              color: 'white',
              padding: '12px',
              borderRadius: '50%',
              cursor: 'pointer'
            }}
          >
            <RefreshCw size={20} />
          </motion.button>
        </header>

        <section style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', 
          gap: '24px',
          marginBottom: '48px'
        }}>
          {[
            { label: 'Active Envs', value: '3', icon: <Zap size={20} />, color: 'var(--accent-math)' },
            { label: 'Avg Reward', value: '0.84', icon: <BarChart3 size={20} />, color: 'var(--accent-negotiate)' },
            { label: 'Uptime', value: '99.9%', icon: <Activity size={20} />, color: 'var(--accent-code)' }
          ].map((stat, idx) => (
            <motion.div 
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="glass-panel" 
              style={{ padding: '24px' }}
            >
              <div style={{ color: stat.color, marginBottom: '12px' }}>{stat.icon}</div>
              <div style={{ fontSize: '2rem', fontWeight: '700', marginBottom: '4px' }}>{stat.value}</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: '500' }}>{stat.label}</div>
            </motion.div>
          ))}
        </section>

        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Environments</h2>
            <div style={{ height: '1px', flex: 1, background: 'var(--border-glass)' }}></div>
          </div>

          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', 
            gap: '32px' 
          }}>
            <AnimatePresence>
              {loading ? (
                Array(3).fill(0).map((_, i) => (
                  <div key={i} className="glass-panel" style={{ height: '300px', opacity: 0.5 }}></div>
                ))
              ) : (
                data?.environments.map((env, idx) => (
                  <motion.div
                    key={env.name}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <EnvironmentCard 
                      env={env} 
                      onClick={() => setSelectedEnv(env)}
                    />
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </section>
      </>
    );
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <aside style={{ 
        width: '280px', 
        borderRight: '1px solid var(--border-glass)',
        padding: '32px 24px',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(5, 5, 5, 0.4)',
        backdropFilter: 'blur(20px)',
        zIndex: 10,
        position: 'sticky',
        top: 0,
        height: '100vh'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '48px' }}>
          <div style={{ 
            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            padding: '10px',
            borderRadius: '12px',
            color: 'white'
          }}>
            <Cpu size={24} />
          </div>
          <span className="brand-text" style={{ fontSize: '1.25rem', fontWeight: '700', letterSpacing: '-0.02em' }}>OpenEnv Suite</span>
        </div>

        <nav style={{ flex: 1 }}>
          <ul style={{ listStyle: 'none' }}>
            <li style={{ marginBottom: '8px' }}>
              <button 
                onClick={() => setActiveTab('dashboard')}
                style={{ 
                  width: '100%',
                  textAlign: 'left',
                  border: 'none',
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  padding: '12px 16px', 
                  borderRadius: '12px',
                  background: activeTab === 'dashboard' ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                  color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontWeight: activeTab === 'dashboard' ? '600' : '500',
                  transition: 'var(--transition-smooth)'
                }}
              >
                <BarChart3 size={20} /> Dashboard
              </button>
            </li>
            <li style={{ marginBottom: '8px' }}>
              <button 
                onClick={() => setActiveTab('monitoring')}
                style={{ 
                  width: '100%',
                  textAlign: 'left',
                  border: 'none',
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  padding: '12px 16px', 
                  borderRadius: '12px',
                  background: activeTab === 'monitoring' ? 'rgba(191, 90, 242, 0.1)' : 'transparent',
                  color: activeTab === 'monitoring' ? 'var(--accent-negotiate)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontWeight: activeTab === 'monitoring' ? '600' : '500',
                  transition: 'var(--transition-smooth)'
                }}
              >
                <Activity size={20} /> Monitoring
              </button>
            </li>
            <li style={{ marginBottom: '8px' }}>
              <button 
                onClick={() => setActiveTab('settings')}
                style={{ 
                  width: '100%',
                  textAlign: 'left',
                  border: 'none',
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  padding: '12px 16px', 
                  borderRadius: '12px',
                  background: activeTab === 'settings' ? 'rgba(50, 215, 75, 0.1)' : 'transparent',
                  color: activeTab === 'settings' ? 'var(--accent-code)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontWeight: activeTab === 'settings' ? '600' : '500',
                  transition: 'var(--transition-smooth)'
                }}
              >
                <Settings size={20} /> Settings
              </button>
            </li>
          </ul>
        </nav>

        <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <Info size={16} className="accent-math" />
            <span style={{ fontSize: '0.85rem', fontWeight: '600' }}>Theme #4</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Self-Improvement via Adaptive Environments</p>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, padding: '48px 64px', overflowY: 'auto' }}>
        {renderContent()}
      </main>

      {/* Details Overlay */}
      <AnimatePresence>
        {selectedEnv && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{ 
              position: 'fixed', 
              inset: 0, 
              background: 'rgba(0,0,0,0.8)', 
              backdropFilter: 'blur(8px)',
              zIndex: 100,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '40px'
            }}
            onClick={() => setSelectedEnv(null)}
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="glass-panel"
              style={{ width: '100%', maxWidth: '800px', padding: '40px', background: '#0a0a0a' }}
              onClick={e => e.stopPropagation()}
            >
              <h2 style={{ fontSize: '2.5rem', marginBottom: '16px' }}>{selectedEnv.name}</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem', marginBottom: '32px' }}>
                {selectedEnv.description}
              </p>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                <div>
                  <h4 style={{ marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.1em', fontSize: '0.8rem', color: 'var(--primary)' }}>Capabilities</h4>
                  <ul style={{ listStyle: 'none' }}>
                    {selectedEnv.tools.map(tool => (
                      <li key={tool} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                        <ChevronRight size={16} className="accent-math" />
                        <code>{tool}</code>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="glass-panel" style={{ padding: '24px', background: 'rgba(255,255,255,0.02)' }}>
                  <h4 style={{ marginBottom: '16px', fontSize: '0.9rem' }}>Quick Actions</h4>
                  <button 
                    className="btn-primary" 
                    style={{ width: '100%', marginBottom: '12px' }}
                    onClick={() => window.open(selectedEnv.docs, '_blank')}
                  >
                    Launch Interactive API
                  </button>
                  <button 
                    style={{ 
                      width: '100%', 
                      background: 'transparent', 
                      border: '1px solid var(--border-glass)',
                      color: 'white',
                      padding: '12px',
                      borderRadius: '12px',
                      cursor: 'pointer'
                    }}
                    onClick={() => window.open(selectedEnv.docs, '_blank')}
                  >
                    View Environment Docs
                  </button>
                  <div style={{ marginTop: '20px', padding: '12px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.8rem', border: '1px dashed var(--border-glass)' }}>
                    <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>Base Endpoint:</div>
                    <code style={{ color: 'var(--accent-math)' }}>{window.location.origin}{selectedEnv.prefix === '/' ? '' : selectedEnv.prefix}</code>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
