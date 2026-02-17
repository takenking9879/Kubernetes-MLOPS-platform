import { ReactFlowProvider } from 'reactflow';
import { MainLayout } from './components/Layout/MainLayout';

export default function App() {
  return (
    <ReactFlowProvider>
      <MainLayout />
    </ReactFlowProvider>
  );
}
