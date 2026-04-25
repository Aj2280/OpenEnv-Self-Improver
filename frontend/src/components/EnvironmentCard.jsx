import React from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, Activity, Terminal, BrainCircuit } from 'lucide-react';

const EnvironmentCard = ({ env, onClick }) => {
  const getIcon = (name) => {
    if (name.includes('Math')) return <Activity className="accent-math" size={24} />;
    if (name.includes('Negotiation')) return <BrainCircuit className="accent-negotiate" size={24} />;
    if (name.includes('Coding')) return <Terminal className="accent-code" size={24} />;
    return <Activity size={24} />;
  };

  const getBadgeClass = (name) => {
    if (name.includes('Math')) return 'badge-math';
    if (name.includes('Negotiation')) return 'badge-negotiate';
    if (name.includes('Coding')) return 'badge-code';
    return '';
  };

  return (
    <motion.div 
      whileHover={{ y: -8, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="glass-panel glow-shadow"
      style={{ padding: '24px', cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
      onClick={onClick}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.05)', 
          padding: '12px', 
          borderRadius: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {getIcon(env.name)}
        </div>
        <span className={`badge ${getBadgeClass(env.name)}`}>
          v{env.version || '1.0'}
        </span>
      </div>

      <h3 style={{ fontSize: '1.5rem', marginBottom: '12px', fontWeight: '600' }}>{env.name}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6', marginBottom: '24px' }}>
        {env.description}
      </p>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '24px' }}>
        {env.tools && env.tools.map(tool => (
          <span key={tool} style={{ 
            fontSize: '0.7rem', 
            background: 'rgba(255,255,255,0.03)', 
            padding: '4px 10px', 
            borderRadius: '6px',
            border: '1px solid rgba(255,255,255,0.05)',
            color: '#aaa'
          }}>
            {tool}
          </span>
        ))}
      </div>

      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        paddingTop: '20px',
        borderTop: '1px solid var(--border-glass)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#32d74b' }}></div>
          <span style={{ fontSize: '0.85rem', color: '#32d74b', fontWeight: '500' }}>Online</span>
        </div>
        <button className="btn-primary" style={{ padding: '8px 16px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
          Details <ExternalLink size={14} />
        </button>
      </div>

      {/* Decorative background glow */}
      <div style={{
        position: 'absolute',
        top: '-10%',
        right: '-10%',
        width: '100px',
        height: '100px',
        background: env.name.includes('Math') ? 'var(--accent-math)' : 
                   env.name.includes('Negotiation') ? 'var(--accent-negotiate)' : 'var(--accent-code)',
        filter: 'blur(80px)',
        opacity: '0.15',
        zIndex: '-1'
      }}></div>
    </motion.div>
  );
};

export default EnvironmentCard;
