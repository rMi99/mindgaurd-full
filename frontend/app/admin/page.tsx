"use client"

import React, { useState, useEffect, createContext, useContext } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, Legend } from 'recharts';

// Helper for class names
const cn = (...classes) => classes.filter(Boolean).join(' ');

// --- ICONS --- //
const Icon = ({ children, className }) => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>{children}</svg>;
const Users = ({ className }) => <Icon className={className}><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /><path d="M16 3.13a4 4 0 0 1 0 7.75" /></Icon>;
const BarChart3 = ({ className }) => <Icon className={className}><path d="M3 3v18h18" /><path d="M18 17V9" /><path d="M13 17V5" /><path d="M8 17v-3" /></Icon>;
const Shield = ({ className }) => <Icon className={className}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></Icon>;
const Flag = ({ className }) => <Icon className={className}><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" /><line x1="4" x2="4" y1="22" y2="15" /></Icon>;
const Trash2 = ({ className }) => <Icon className={className}><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /><line x1="10" x2="10" y1="11" y2="17" /><line x1="14" x2="14" y1="11" y2="17" /></Icon>;
const Eye = ({ className }) => <Icon className={className}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></Icon>;
const TrendingUp = ({ className }) => <Icon className={className}><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></Icon>;
const Activity = ({ className }) => <Icon className={className}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></Icon>;
const X = ({ className }) => <Icon className={className}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></Icon>;

// --- UI COMPONENTS --- //
const Card = ({ className, ...props }) => <div className={cn("rounded-xl border bg-card text-card-foreground shadow-sm", className)} {...props} />;
const CardHeader = ({ className, ...props }) => <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />;
const CardTitle = ({ className, ...props }) => <h3 className={cn("text-lg font-semibold leading-none tracking-tight", className)} {...props} />;
const CardDescription = ({ className, ...props }) => <p className={cn("text-sm text-muted-foreground", className)} {...props} />;
const CardContent = ({ className, ...props }) => <div className={cn("p-6 pt-0", className)} {...props} />;

const Button = ({ className, variant, size, ...props }) => {
    const variants = { default: "bg-blue-600 text-white hover:bg-blue-700", destructive: "bg-red-600 text-white hover:bg-red-700", outline: "border border-input bg-transparent hover:bg-accent hover:text-accent-foreground", secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80" };
    const sizes = { default: "h-10 px-4 py-2", sm: "h-9 rounded-md px-3", lg: "h-11 rounded-md px-8" };
    return <button className={cn("inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none", variants[variant] || variants.default, sizes[size] || sizes.default, className)} {...props} />;
};

const Badge = ({ className, variant, ...props }) => {
    const variants = { default: "border-transparent bg-blue-600 text-white", secondary: "border-transparent bg-gray-200 text-gray-800", destructive: "border-transparent bg-red-600 text-white" };
    return <div className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors", variants[variant] || variants.default, className)} {...props} />;
};

const TabsContext = createContext();
const Tabs = ({ defaultValue, children, className }) => {
    const [activeTab, setActiveTab] = useState(defaultValue);
    return <TabsContext.Provider value={{ activeTab, setActiveTab }}><div className={className}>{children}</div></TabsContext.Provider>;
}
const TabsList = ({ className, children }) => <div className={cn("inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground", className)}>{children}</div>;
const TabsTrigger = ({ value, children, className }) => {
    const { activeTab, setActiveTab } = useContext(TabsContext);
    const isActive = activeTab === value;
    return <button onClick={() => setActiveTab(value)} className={cn("inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50", isActive ? "bg-background text-foreground shadow-sm" : "hover:bg-gray-100", className)}>{children}</button>;
};
const TabsContent = ({ value, children, className }) => {
    const { activeTab } = useContext(TabsContext);
    return activeTab === value ? <div className={cn("mt-2", className)}>{children}</div> : null;
};

const Input = ({ className, ...props }) => <input className={cn("flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50", className)} {...props} />;
const Label = ({ className, ...props }) => <label className={cn("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70", className)} {...props} />;
const Textarea = ({ className, ...props }) => <textarea className={cn("flex min-h-[80px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50", className)} {...props} />;

const SelectContext = React.createContext();
const Select = ({ children, value, onValueChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    return <SelectContext.Provider value={{ isOpen, setIsOpen, value, onValueChange }}><div className="relative">{children}</div></SelectContext.Provider>;
}
const SelectTrigger = ({ children, className }) => {
    const { setIsOpen } = useContext(SelectContext);
    return <button onClick={() => setIsOpen(o => !o)} className={cn("flex h-10 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50", className)}>{children}</button>;
}
const SelectValue = () => {
    const { value } = useContext(SelectContext);
    return <span>{value}</span>;
}
const SelectContent = ({ children, className }) => {
    const { isOpen } = useContext(SelectContext);
    if (!isOpen) return null;
    return <div className={cn("absolute z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md animate-in fade-in-80", className)}>{children}</div>;
}
const SelectItem = ({ value, children, className }) => {
    const { onValueChange, setIsOpen } = useContext(SelectContext);
    return <div onClick={() => { onValueChange(value); setIsOpen(false); }} className={cn("relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50", className)}>{children}</div>;
}

const DialogContext = createContext();
const Dialog = ({ children }) => {
    const [isOpen, setIsOpen] = useState(false);
    return <DialogContext.Provider value={{ isOpen, setIsOpen }}>{children}</DialogContext.Provider>;
};
const DialogTrigger = ({ children }) => {
    const { setIsOpen } = useContext(DialogContext);
    return <div onClick={() => setIsOpen(true)}>{children}</div>;
};
const DialogContent = ({ children, className }) => {
    const { isOpen, setIsOpen } = useContext(DialogContext);
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
            <div className={cn("relative m-4 w-full max-w-lg rounded-xl border bg-background p-6 shadow-lg", className)}>
                <button onClick={() => setIsOpen(false)} className="absolute top-4 right-4"><X className="w-4 h-4" /></button>
                {children}
            </div>
        </div>
    );
};
const DialogHeader = ({ className, ...props }) => <div className={cn("flex flex-col space-y-1.5 text-center sm:text-left", className)} {...props} />;
const DialogTitle = ({ className, ...props }) => <h2 className={cn("text-lg font-semibold leading-none tracking-tight", className)} {...props} />;
const DialogDescription = ({ className, ...props }) => <p className={cn("text-sm text-muted-foreground", className)} {...props} />;

const AlertDialog = Dialog;
const AlertDialogTrigger = DialogTrigger;
const AlertDialogContent = DialogContent;
const AlertDialogHeader = DialogHeader;
const AlertDialogTitle = DialogTitle;
const AlertDialogDescription = DialogDescription;
const AlertDialogFooter = ({ className, ...props }) => <div className={cn("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 pt-4", className)} {...props} />;
const AlertDialogCancel = (props) => {
    const { setIsOpen } = useContext(DialogContext);
    return <Button variant="outline" onClick={() => setIsOpen(false)} {...props} />;
};
const AlertDialogAction = (props) => {
    const { setIsOpen } = useContext(DialogContext);
    const handleClick = (e) => {
        if (props.onClick) props.onClick(e);
        setIsOpen(false);
    };
    return <Button variant="destructive" {...props} onClick={handleClick} />;
};

const Table = ({ className, ...props }) => <div className="relative w-full overflow-auto"><table className={cn("w-full caption-bottom text-sm", className)} {...props} /></div>;
const TableHeader = ({ className, ...props }) => <thead className={cn("[&_tr]:border-b", className)} {...props} />;
const TableBody = ({ className, ...props }) => <tbody className={cn("[&_tr:last-child]:border-0", className)} {...props} />;
const TableRow = ({ className, ...props }) => <tr className={cn("border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted", className)} {...props} />;
const TableHead = ({ className, ...props }) => <th className={cn("h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0", className)} {...props} />;
const TableCell = ({ className, ...props }) => <td className={cn("p-4 align-middle [&:has([role=checkbox])]:pr-0", className)} {...props} />;

// --- MAIN ADMIN PAGE COMPONENT --- //

export default function App() {
  const [adminToken, setAdminToken] = useState("")
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [systemStats, setSystemStats] = useState(null)
  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const [flagReason, setFlagReason] = useState("");
  const [flagPriority, setFlagPriority] = useState("medium");

  const authenticate = async () => {
    setError(null);
    if (adminToken === "123") {
      setIsAuthenticated(true);
      // Don't fetch data here, it will be fetched in useEffect
    } else {
      setError("Invalid admin token");
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      fetchSystemStats();
      fetchUsers();
    }
  }, [isAuthenticated]);

  const fetchSystemStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/admin/stats?token=${adminToken}`);
      if (!response.ok) throw new Error('Failed to fetch system stats');
      const data = await response.json();
      setSystemStats(data);
    } catch (error) {
      console.error("Failed to fetch system stats:", error);
      setError("Could not load system statistics.");
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/admin/users?token=${adminToken}&limit=100`);
      if (!response.ok) throw new Error('Failed to fetch users');
      const data = await response.json();
      setUsers(data);
    } catch (error) {
      console.error("Failed to fetch users:", error);
      setError("Could not load user data.");
    } finally {
      setLoading(false);
    }
  };

  const fetchUserDetail = async (userId) => {
    setSelectedUser(null);
    try {
      const response = await fetch(`/api/admin/users/${userId}?token=${adminToken}`);
      if (!response.ok) throw new Error('Failed to fetch user details');
      const data = await response.json();
      setSelectedUser(data);
    } catch (error) {
      console.error("Failed to fetch user detail:", error);
      setError(`Could not load details for user ${userId}.`);
    }
  };

  const deleteUser = async (userId, reason) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}?token=${adminToken}&reason=${encodeURIComponent(reason)}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete user');
      await fetchUsers(); // Refresh user list
      if (selectedUser && selectedUser.user_id === userId) {
        setSelectedUser(null);
      }
    } catch (error) {
      console.error("Failed to delete user:", error);
      setError(`Failed to delete user ${userId}.`);
    }
  };

  const flagUser = async (userId, reason, priority) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/flag?token=${adminToken}&reason=${encodeURIComponent(reason)}&priority=${priority}`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to flag user');
      await fetchUsers(); // Refresh user list to show flag status
    } catch (error) {
      console.error("Failed to flag user:", error);
      setError(`Failed to flag user ${userId}.`);
    }
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case "low": return "bg-green-100 text-green-800";
      case "moderate": return "bg-yellow-100 text-yellow-800";
      case "high": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };
  
  const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#6b7280'];
  const riskData = systemStats?.risk_level_distribution ? Object.entries(systemStats.risk_level_distribution).map(([name, value]) => ({ name, value })) : [];
  const dailyData = systemStats?.daily_assessment_counts ? Object.entries(systemStats.daily_assessment_counts).map(([date, count]) => ({ date, count })) : [];


  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <Shield className="w-6 h-6 text-blue-600" />
              Admin Authentication
            </CardTitle>
            <CardDescription>Enter admin token to access the management panel.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="token">Admin Token</Label>
              <Input
                id="token"
                type="password"
                value={adminToken}
                onChange={(e) => setAdminToken(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && authenticate()}
                placeholder="Enter admin token"
              />
            </div>
            {error && <div className="text-red-600 text-sm font-medium">{error}</div>}
            <Button onClick={authenticate} className="w-full">Authenticate</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <header className="bg-white shadow-sm sticky top-0 z-40">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center"><Shield className="w-6 h-6 text-white" /></div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Admin Panel</h1>
                <p className="text-sm text-gray-500">MindGuard System Management</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="destructive">Admin Access</Badge>
              <Button variant="outline" onClick={() => { setIsAuthenticated(false); setAdminToken(''); setError(null); }}>Logout</Button>
            </div>
          </div>
        </div>
      </header>
      
      <main className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="users">User Management</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="flags">Flagged Users</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            {loading && <p>Loading dashboard...</p>}
            {error && <p className="text-red-500">{error}</p>}
            {systemStats && (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card><CardHeader className="flex flex-row items-center justify-between pb-2"><CardTitle className="text-sm font-medium">Total Users</CardTitle><Users className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{systemStats.total_users?.toLocaleString()}</div></CardContent></Card>
                  <Card><CardHeader className="flex flex-row items-center justify-between pb-2"><CardTitle className="text-sm font-medium">Total Assessments</CardTitle><BarChart3 className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{systemStats.total_assessments?.toLocaleString()}</div></CardContent></Card>
                  <Card><CardHeader className="flex flex-row items-center justify-between pb-2"><CardTitle className="text-sm font-medium">Active Users (30d)</CardTitle><Activity className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{systemStats.active_users_last_30_days?.toLocaleString()}</div></CardContent></Card>
                  <Card><CardHeader className="flex flex-row items-center justify-between pb-2"><CardTitle className="text-sm font-medium">Avg Assess/User</CardTitle><TrendingUp className="h-4 w-4 text-muted-foreground" /></CardHeader><CardContent><div className="text-2xl font-bold">{systemStats.average_assessments_per_user?.toFixed(2)}</div></CardContent></Card>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                    <Card className="lg:col-span-2">
                        <CardHeader><CardTitle>Risk Level Distribution</CardTitle><CardDescription>Current distribution of user risk levels.</CardDescription></CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie data={riskData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                                        {riskData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                    </Pie>
                                    <Tooltip />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                    <Card className="lg:col-span-3">
                        <CardHeader><CardTitle>Daily Assessment Activity</CardTitle><CardDescription>Assessment counts over the last 30 days.</CardDescription></CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={dailyData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="date" tickFormatter={(tick) => new Date(tick).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} />
                                    <YAxis />
                                    <Tooltip labelFormatter={(label) => new Date(label).toLocaleDateString()} />
                                    <Bar dataKey="count" fill="#3b82f6" />
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </div>
              </>
            )}
          </TabsContent>
          
          <TabsContent value="users" className="space-y-6">
            <Card>
              <CardHeader><CardTitle>User Management</CardTitle><CardDescription>Manage user accounts and view detailed information.</CardDescription></CardHeader>
              <CardContent>
                {loading ? <div className="text-center py-8">Loading users...</div> : (
                  <Table>
                    <TableHeader><TableRow><TableHead>User ID</TableHead><TableHead>Assessments</TableHead><TableHead>Last Activity</TableHead><TableHead>Risk Level</TableHead><TableHead>Status</TableHead><TableHead>Actions</TableHead></TableRow></TableHeader>
                    <TableBody>
                      {users.map((user) => (
                        <TableRow key={user.user_id} className={user.is_flagged ? 'bg-orange-50' : ''}>
                          <TableCell className="font-mono text-sm">{user.user_id}</TableCell>
                          <TableCell>{user.total_assessments}</TableCell>
                          <TableCell>{user.last_assessment_date ? new Date(user.last_assessment_date).toLocaleDateString() : 'N/A'}</TableCell>
                          <TableCell><Badge className={getRiskLevelColor(user.current_risk_level)}>{user.current_risk_level || 'N/A'}</Badge></TableCell>
                          <TableCell><Badge variant={user.is_active ? "default" : "secondary"}>{user.is_active ? "Active" : "Inactive"}</Badge></TableCell>
                          <TableCell>
                            <div className="flex space-x-2">
                                <Button variant="outline" size="sm" onClick={() => fetchUserDetail(user.user_id)}><Eye className="w-4 h-4" /></Button>
                                <Dialog>
                                    <DialogTrigger asChild><Button variant="outline" size="sm"><Flag className="w-4 h-4" /></Button></DialogTrigger>
                                    <DialogContent>
                                        <DialogHeader><DialogTitle>Flag User: {user.user_id}</DialogTitle><DialogDescription>Flag this user for manual review.</DialogDescription></DialogHeader>
                                        <div className="space-y-4 py-4">
                                            <div><Label htmlFor="flag-reason">Reason</Label><Textarea id="flag-reason" placeholder="Enter reason for flagging..." onChange={(e) => setFlagReason(e.target.value)} /></div>
                                            <div><Label>Priority</Label>
                                                <Select value={flagPriority} onValueChange={setFlagPriority}>
                                                    <SelectTrigger><SelectValue /></SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="low">Low</SelectItem>
                                                        <SelectItem value="medium">Medium</SelectItem>
                                                        <SelectItem value="high">High</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <Button onClick={() => flagUser(user.user_id, flagReason, flagPriority)}>Flag User</Button>
                                        </div>
                                    </DialogContent>
                                </Dialog>
                                <AlertDialog>
                                    <AlertDialogTrigger asChild><Button variant="destructive" size="sm"><Trash2 className="w-4 h-4" /></Button></AlertDialogTrigger>
                                    <AlertDialogContent>
                                        <AlertDialogHeader><AlertDialogTitle>Are you sure?</AlertDialogTitle><AlertDialogDescription>This will permanently delete the user account for {user.user_id} and all associated data.</AlertDialogDescription></AlertDialogHeader>
                                        <AlertDialogFooter><AlertDialogCancel>Cancel</AlertDialogCancel><AlertDialogAction onClick={() => deleteUser(user.user_id, "Admin deletion")}>Delete</AlertDialogAction></AlertDialogFooter>
                                    </AlertDialogContent>
                                </AlertDialog>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
            {selectedUser && (
              <Card>
                <CardHeader><CardTitle>User Details: {selectedUser.user_id}</CardTitle><CardDescription>Complete user information and assessment history.</CardDescription></CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-2">Account Information</h4>
                      <div className="space-y-2 text-sm">
                        <p><strong>User ID:</strong> {selectedUser.user_id}</p>
                        <p><strong>Total Assessments:</strong> {selectedUser.statistics?.total_assessments}</p>
                        <p><strong>Average PHQ-9:</strong> {selectedUser.statistics?.average_phq9_score}</p>
                        <p><strong>Account Status:</strong> {selectedUser.account_status}</p>
                        <p><strong>Last Activity:</strong> {selectedUser.last_activity ? new Date(selectedUser.last_activity).toLocaleString() : 'N/A'}</p>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Recent Assessments</h4>
                      <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
                        {selectedUser.assessments?.slice(0, 10).map((assessment) => (
                          <div key={assessment.id} className="text-sm border rounded p-2">
                            <p><strong>Date:</strong> {new Date(assessment.date).toLocaleDateString()}</p>
                            <p><strong>PHQ-9:</strong> {assessment.phq9_score}/27</p>
                            <p><strong>Risk:</strong> <span className={`font-bold ${getRiskLevelColor(assessment.risk_level).split(' ')[1]}`}>{assessment.risk_level}</span></p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
          
          <TabsContent value="analytics"><Card><CardHeader><CardTitle>System Analytics</CardTitle><CardDescription>This is a placeholder for more advanced analytics.</CardDescription></CardHeader><CardContent><p>Advanced analytics features would be implemented here.</p></CardContent></Card></TabsContent>
          
          <TabsContent value="flags">
              <Card>
                  <CardHeader><CardTitle>Flagged Users</CardTitle><CardDescription>Users that have been flagged for manual review.</CardDescription></CardHeader>
                  <CardContent>
                      <Table>
                          <TableHeader><TableRow><TableHead>User ID</TableHead><TableHead>Risk Level</TableHead><TableHead>Flagged On</TableHead><TableHead>Actions</TableHead></TableRow></TableHeader>
                          <TableBody>
                              {users.filter(u => u.is_flagged).map(user => (
                                  <TableRow key={user.user_id}>
                                      <TableCell>{user.user_id}</TableCell>
                                      <TableCell><Badge className={getRiskLevelColor(user.current_risk_level)}>{user.current_risk_level}</Badge></TableCell>
                                      <TableCell>{new Date().toLocaleDateString()}</TableCell>
                                      <TableCell><Button variant="outline" size="sm">Review</Button></TableCell>
                                  </TableRow>
                              ))}
                          </TableBody>
                      </Table>
                  </CardContent>
              </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
