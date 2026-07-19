import { SidebarInset } from "@/components/ui/sidebar";
import AppSidebar from "@/components/app-sidebar";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <AppSidebar />
      <SidebarInset>
        {children}
      </SidebarInset>
    </>
  );
}
