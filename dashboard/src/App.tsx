import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { OverviewPage } from "@/pages/Overview";
import { TrainingPage } from "@/pages/Training";
import { EvaluationPage } from "@/pages/Evaluation";
import { FindingsPage } from "@/pages/Findings";
import { CrossGamePage } from "@/pages/CrossGame";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/overview" replace />} />
        <Route path="/overview" element={<OverviewPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/findings" element={<FindingsPage />} />
        <Route path="/cross-game" element={<CrossGamePage />} />
      </Route>
    </Routes>
  );
}
