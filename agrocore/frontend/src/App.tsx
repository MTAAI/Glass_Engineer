import { Routes, Route } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { LoginPage } from "@/pages/Login";
import { DashboardPage } from "@/pages/Dashboard";
import { FarmPage } from "@/pages/Farm";
import { LivestockPage } from "@/pages/Livestock";
import { InventoryPage } from "@/pages/Inventory";
import {
  MapPage,
  AquaculturePage,
  ProcessingPage,
  KitchenPage,
  MenuPage,
  HACCPPage,
  MachineryPage,
  WorkersPage,
  CostPricePage,
  ReportsPage,
  CameraPage,
  SettingsPage,
} from "@/pages/placeholders";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<DashboardPage />} />
        <Route path="map" element={<MapPage />} />
        <Route path="farm" element={<FarmPage />} />
        <Route path="livestock" element={<LivestockPage />} />
        <Route path="aquaculture" element={<AquaculturePage />} />
        <Route path="processing" element={<ProcessingPage />} />
        <Route path="kitchen" element={<KitchenPage />} />
        <Route path="menu" element={<MenuPage />} />
        <Route path="haccp" element={<HACCPPage />} />
        <Route path="machinery" element={<MachineryPage />} />
        <Route path="workers" element={<WorkersPage />} />
        <Route path="inventory" element={<InventoryPage />} />
        <Route path="cost" element={<CostPricePage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="camera" element={<CameraPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
