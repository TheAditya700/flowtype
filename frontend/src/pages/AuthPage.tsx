import React from 'react';
import AuthModal from '../components/AuthModal';

const AuthPage: React.FC = () => {
  return (
    <div className="w-full max-w-md mx-auto p-8 flex items-center justify-center min-h-[60vh]">
      <AuthModal showModal={true} onClose={() => {}} isPage={true} />
    </div>
  );
};

export default AuthPage;
