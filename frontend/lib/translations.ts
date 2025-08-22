export type Language = 'en' | 'si' | 'ta' | 'es' | 'fr' | 'zh';

export interface TranslationKeys {
  // Common
  title: string;
  subtitle: string;
  privacyNotice: string;
  loading: string;
  error: string;
  success: string;
  cancel: string;
  save: string;
  delete: string;
  edit: string;
  add: string;
  search: string;
  filter: string;
  sort: string;
  next: string;
  previous: string;
  submit: string;
  back: string;
  close: string;
  
  // Navigation
  dashboard: string;
  assessment: string;
  history: string;
  research: string;
  profile: string;
  settings: string;
  logout: string;
  login: string;
  register: string;
  
  // Assessment
  healthAssessment: string;
  basicInformation: string;
  physicalHealth: string;
  mentalWellbeing: string;
  lifestyleFactors: string;
  reviewSubmit: string;
  fullName: string;
  age: string;
  gender: string;
  occupation: string;
  email: string;
  sleepHours: string;
  exerciseFrequency: string;
  dietQuality: string;
  stressLevel: string;
  moodScore: string;
  mentalHealthHistory: string;
  socialConnections: string;
  workLifeBalance: string;
  financialStress: string;
  additionalNotes: string;
  
  // PHQ-9 Screening
  mentalHealthScreening: string;
  overLastTwoWeeks: string;
  notAtAll: string;
  severalDays: string;
  moreThanHalf: string;
  nearlyEveryDay: string;
  
  // Dashboard
  overallHealthScore: string;
  riskLevel: string;
  recommendations: string;
  brainHealActivities: string;
  weeklyPlan: string;
  progressTracking: string;
  
  // Risk Levels
  low: string;
  normal: string;
  high: string;
  excellent: string;
  good: string;
  fair: string;
  needsAttention: string;
  
  // Health Metrics
  sleep: string;
  exercise: string;
  stress: string;
  mood: string;
  energy: string;
  social: string;
  diet: string;
  
  // Time Ranges
  days7: string;
  days30: string;
  days90: string;
  year1: string;
  
  // Chart Types
  trendAnalysis: string;
  wellnessOverview: string;
  metricComparison: string;
  healthRadar: string;
  comprehensiveView: string;
  
  // Messages
  welcomeMessage: string;
  assessmentComplete: string;
  dataSaved: string;
  errorOccurred: string;
  noDataAvailable: string;
  
  // Validation
  required: string;
  invalidEmail: string;
  invalidAge: string;
  passwordTooShort: string;
  
  // Actions
  startAssessment: string;
  continueAssessment: string;
  viewResults: string;
  downloadReport: string;
  shareResults: string;

  welcome: string
  assessmentComplete: string
  riskLevel: string
  startNewAssessment: string
  viewDetailedReport: string
}

const translations: Record<Language, TranslationKeys> = {
  en: {
    // Common
    title: 'MindGuard',
    subtitle: 'AI-Powered Health Risk Prediction',
    privacyNotice: "Your privacy is important to us. We do not share your data with third parties.",
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    add: 'Add',
    search: 'Search',
    filter: 'Filter',
    sort: 'Sort',
    next: 'Next',
    previous: 'Previous',
    submit: 'Submit',
    back: 'Back',
    close: 'Close',
    
    // Navigation
    dashboard: 'Dashboard',
    assessment: 'Assessment',
    history: 'History',
    research: 'Research',
    profile: 'Profile',
    settings: 'Settings',
    logout: 'Logout',
    login: 'Login',
    register: 'Register',
    
    // Assessment
    healthAssessment: 'Health Assessment',
    basicInformation: 'Basic Information',
    physicalHealth: 'Physical Health',
    mentalWellbeing: 'Mental Wellbeing',
    lifestyleFactors: 'Lifestyle Factors',
    reviewSubmit: 'Review & Submit',
    fullName: 'Full Name',
    age: 'Age',
    gender: 'Gender',
    occupation: 'Occupation',
    email: 'Email Address',
    sleepHours: 'Sleep Hours',
    exerciseFrequency: 'Exercise Frequency',
    dietQuality: 'Diet Quality',
    stressLevel: 'Stress Level',
    moodScore: 'Mood Score',
    mentalHealthHistory: 'Mental Health History',
    socialConnections: 'Social Connections',
    workLifeBalance: 'Work-Life Balance',
    financialStress: 'Financial Stress',
    additionalNotes: 'Additional Notes',
    
    // PHQ-9 Screening
    mentalHealthScreening: 'Mental Health Screening',
    overLastTwoWeeks: 'Over the last 2 weeks, how often have you been bothered by any of the following problems?',
    notAtAll: 'Not at all',
    severalDays: 'Several days',
    moreThanHalf: 'More than half the days',
    nearlyEveryDay: 'Nearly every day',
    
    // Dashboard
    overallHealthScore: 'Overall Health Score',
    riskLevel: 'Risk Level',
    recommendations: 'Recommendations',
    brainHealActivities: 'Brain Heal Activities',
    weeklyPlan: 'Weekly Plan',
    progressTracking: 'Progress Tracking',
    
    // Risk Levels
    low: 'Low',
    normal: 'Normal',
    high: 'High',
    excellent: 'Excellent',
    good: 'Good',
    fair: 'Fair',
    needsAttention: 'Needs Attention',
    
    // Health Metrics
    sleep: 'Sleep',
    exercise: 'Exercise',
    stress: 'Stress',
    mood: 'Mood',
    energy: 'Energy',
    social: 'Social',
    diet: 'Diet',
    
    // Time Ranges
    days7: '7 Days',
    days30: '30 Days',
    days90: '90 Days',
    year1: '1 Year',
    
    // Chart Types
    trendAnalysis: 'Trend Analysis',
    wellnessOverview: 'Wellness Overview',
    metricComparison: 'Metric Comparison',
    healthRadar: 'Health Radar',
    comprehensiveView: 'Comprehensive View',
    
    // Messages
    welcomeMessage: 'Welcome to MindGuard',
    assessmentComplete: 'Assessment Complete',
    dataSaved: 'Data saved successfully',
    errorOccurred: 'An error occurred',
    noDataAvailable: 'No data available',
    
    // Validation
    required: 'This field is required',
    invalidEmail: 'Please enter a valid email address',
    invalidAge: 'Please enter a valid age between 13 and 120',
    passwordTooShort: 'Password must be at least 8 characters long',
    
    // Actions
    startAssessment: 'Start Assessment',
    continueAssessment: 'Continue Assessment',
    viewResults: 'View Results',
    downloadReport: 'Download Report',
    shareResults: 'Share Results',
  },
  
  si: {
    // Common
    title: 'MindGuard',
    subtitle: 'කෘතිම බුද්ධිය මත පදනම් වූ සෞඛ්‍ය අවදානම් අනාවැකිය',
    loading: 'පූරණය වෙමින්...',
    error: 'දෝෂයකි',
    success: 'සාර්ථකයි',
    cancel: 'අවලංගු කරන්න',
    save: 'සුරකින්න',
    delete: 'මකන්න',
    edit: 'සංස්කරණය කරන්න',
    add: 'එකතු කරන්න',
    search: 'සොයන්න',
    filter: 'පෙරහන් කරන්න',
    sort: 'වර්ග කරන්න',
    next: 'ඊළඟ',
    previous: 'පෙර',
    submit: 'ඉදිරිපත් කරන්න',
    back: 'ආපසු',
    close: 'වසන්න',
    
    // Navigation
    dashboard: 'පාලන පුවරුව',
    assessment: 'තක්සේරුව',
    history: 'ඉතිහාසය',
    research: 'පර්යේෂණ',
    profile: 'පැතිකඩ',
    settings: 'සැකසුම්',
    logout: 'පිටවන්න',
    login: 'ඇතුල් වන්න',
    register: 'ලියාපදිංචි වන්න',
    
    // Assessment
    healthAssessment: 'සෞඛ්‍ය තක්සේරුව',
    basicInformation: 'මූලික තොරතුරු',
    physicalHealth: 'ශාරීරික සෞඛ්‍යය',
    mentalWellbeing: 'මානසික යහපැවැත්ම',
    lifestyleFactors: 'ජීවන රටා සාධක',
    reviewSubmit: 'සමාලෝචනය කර ඉදිරිපත් කරන්න',
    fullName: 'සම්පූර්ණ නම',
    age: 'වයස',
    gender: 'ස්ත්‍රී පුරුෂ භාවය',
    occupation: 'රැකියාව',
    email: 'විද්‍යුත් තැපැල් ලිපිනය',
    sleepHours: 'නින්දේ පැය ගණන',
    exerciseFrequency: 'ව්‍යායාම සංඛ්‍යාතය',
    dietQuality: 'ආහාරයේ ගුණාත්මකභාවය',
    stressLevel: 'මානසික ආතතිය',
    moodScore: 'මනෝභාව ලකුණු',
    mentalHealthHistory: 'මානසික සෞඛ්‍ය ඉතිහාසය',
    socialConnections: 'සමාජ සම්බන්ධතා',
    workLifeBalance: 'වැඩ-ජීවිත සමබරතාවය',
    financialStress: 'මූල්‍ය ආතතිය',
    additionalNotes: 'අමතර සටහන්',
    
    // PHQ-9 Screening
    mentalHealthScreening: 'මානසික සෞඛ්‍ය පරීක්ෂණය',
    overLastTwoWeeks: 'පසුගිය සති 2 තුළ, මෙම ගැටලු මගින් ඔබ කොපමණ කරදරයට පත් වී ඇත?',
    notAtAll: 'කිසිසේත්ම නොවේ',
    severalDays: 'දින කිහිපයක්',
    moreThanHalf: 'දින අඩකට වඩා',
    nearlyEveryDay: 'සෑම දිනකම පාහේ',
    
    // Dashboard
    overallHealthScore: 'සමස්ත සෞඛ්‍ය ලකුණු',
    riskLevel: 'අවදානම් මට්ටම',
    recommendations: 'නිර්දේශ',
    brainHealActivities: 'මොළය සුවපත් කිරීමේ ක්‍රියාකාරකම්',
    weeklyPlan: 'සතිපතා සැලැස්ම',
    progressTracking: 'ප්‍රගතිය නිරීක්ෂණය',
    
    // Risk Levels
    low: 'අඩු',
    normal: 'සාමාන්‍ය',
    high: 'ඉහළ',
    excellent: 'විශිෂ්ට',
    good: 'හොඳ',
    fair: 'සාධාරණ',
    needsAttention: 'අවධානය අවශ්‍යයි',
    
    // Health Metrics
    sleep: 'නින්ද',
    exercise: 'ව්‍යායාම',
    stress: 'ආතතිය',
    mood: 'මනෝභාවය',
    energy: 'ශක්තිය',
    social: 'සමාජ',
    diet: 'ආහාර',
    
    // Time Ranges
    days7: 'දින 7',
    days30: 'දින 30',
    days90: 'දින 90',
    year1: 'වසර 1',
    
    // Chart Types
    trendAnalysis: 'ප්‍රවණතා විශ්ලේෂණය',
    wellnessOverview: 'යහපැවැත්ම පිළිබඳ දළ විශ්ලේෂණය',
    metricComparison: 'මෙට්‍රික් සැසඳීම',
    healthRadar: 'සෞඛ්‍ය රේඩාර්',
    comprehensiveView: 'පුළුල් දැක්ම',
    
    // Messages
    welcomeMessage: 'MindGuard වෙත සාදරයෙන් පිළිගනිමු',
    assessmentComplete: 'තක්සේරුව සම්පූර්ණයි',
    dataSaved: 'දත්ත සාර්ථකව සුරකින ලදි',
    errorOccurred: 'දෝෂයක් ඇතිවිය',
    noDataAvailable: 'දත්ත නොමැත',
    
    // Validation
    required: 'මෙම ක්ෂේත්‍රය අනිවාර්ය වේ',
    invalidEmail: 'කරුණාකර වලංගු විද්‍යුත් තැපැල් ලිපිනයක් ඇතුළත් කරන්න',
    invalidAge: 'කරුණාකර වයස 13 සහ 120 අතර වලංගු වයසක් ඇතුළත් කරන්න',
    passwordTooShort: 'මුරපදය අවම වශයෙන් අක්ෂර 8 ක් දිග විය යුතුය',
    
    // Actions
    startAssessment: 'තක්සේරුව ආරම්භ කරන්න',
    continueAssessment: 'තක්සේරුව දිගටම කරගෙන යන්න',
    viewResults: 'ප්‍රතිඵල බලන්න',
    downloadReport: 'වාර්තාව බාගන්න',
    shareResults: 'ප්‍රතිඵල බෙදාගන්න',
  },
  
  ta: {
    // Common
    title: 'MindGuard',
    subtitle: 'செயற்கை நுண்ணறிவு ఆధారిత சுகாதார இடர் முன்கணிப்பு',
    loading: 'ஏற்றுகிறது...',
    error: 'பிழை',
    success: 'வெற்றி',
    cancel: 'ரத்துசெய்',
    save: 'சேமி',
    delete: 'நீக்கு',
    edit: 'திருத்து',
    add: 'சேர்',
    search: 'தேடு',
    filter: 'வடிகட்டு',
    sort: 'வரிசைப்படுத்து',
    next: 'அடுத்து',
    previous: 'முந்தைய',
    submit: 'சமர்ப்பி',
    back: 'பின்செல்',
    close: 'மூடு',
    
    // Navigation
    dashboard: 'கட்டுப்பாட்டு பலகம்',
    assessment: 'மதிப்பீடு',
    history: 'வரலாறு',
    research: 'ஆராய்ச்சி',
    profile: 'சுயவிவரம்',
    settings: 'அமைப்புகள்',
    logout: 'வெளியேறு',
    login: 'உள்நுழை',
    register: 'பதிவு செய்',
    
    // Assessment
    healthAssessment: 'சுகாதார மதிப்பீடு',
    basicInformation: 'அடிப்படைத் தகவல்',
    physicalHealth: 'உடல்நலம்',
    mentalWellbeing: 'மன நலம்',
    lifestyleFactors: 'வாழ்க்கை முறை காரணிகள்',
    reviewSubmit: 'மதிப்பாய்வு செய்து சமர்ப்பிக்கவும்',
    fullName: 'முழு பெயர்',
    age: 'வயது',
    gender: 'பாலினம்',
    occupation: 'தொழில்',
    email: 'மின்னஞ்சல் முகவரி',
    sleepHours: 'தூக்க நேரம்',
    exerciseFrequency: 'உடற்பயிற்சி அதிர்வெண்',
    dietQuality: 'உணவின் தரம்',
    stressLevel: 'மன அழுத்த நிலை',
    moodScore: 'மனநிலை மதிப்பெண்',
    mentalHealthHistory: 'மனநல வரலாறு',
    socialConnections: 'சமூகத் தொடர்புகள்',
    workLifeBalance: 'வேலை-வாழ்க்கைச் சமநிலை',
    financialStress: 'நிதி அழுத்தம்',
    additionalNotes: 'கூடுதல் குறிப்புகள்',
    
    // PHQ-9 Screening
    mentalHealthScreening: 'மனநலப் பரிசோதனை',
    overLastTwoWeeks: 'கடந்த 2 வாரங்களில், இந்தப் பிரச்சினைகளால் நீங்கள் எவ்வளவு அதிகம் பாதிக்கப்பட்டீர்கள்?',
    notAtAll: 'இல்லவே இல்லை',
    severalDays: 'சில நாட்கள்',
    moreThanHalf: 'பாதி நாட்களுக்கு மேல்',
    nearlyEveryDay: 'ஏறத்தாழ தினமும்',
    
    // Dashboard
    overallHealthScore: 'ஒட்டுமொத்த சுகாதார மதிப்பெண்',
    riskLevel: 'இடர் நிலை',
    recommendations: 'பரிந்துரைகள்',
    brainHealActivities: 'மூளை குணப்படுத்தும் செயல்பாடுகள்',
    weeklyPlan: 'வாராந்திரத் திட்டம்',
    progressTracking: 'முன்னேற்றத்தைக் கண்காணித்தல்',
    
    // Risk Levels
    low: 'குறைந்த',
    normal: 'சாதாரண',
    high: 'அதிக',
    excellent: 'சிறந்த',
    good: 'நல்ல',
    fair: 'சுமார்',
    needsAttention: 'கவனம் தேவை',
    
    // Health Metrics
    sleep: 'தூக்கம்',
    exercise: 'உடற்பயிற்சி',
    stress: 'மன அழுத்தம்',
    mood: 'மனநிலை',
    energy: 'ஆற்றல்',
    social: 'சமூகம்',
    diet: 'உணவு',
    
    // Time Ranges
    days7: '7 நாட்கள்',
    days30: '30 நாட்கள்',
    days90: '90 நாட்கள்',
    year1: '1 ஆண்டு',
    
    // Chart Types
    trendAnalysis: 'போக்கு பகுப்பாய்வு',
    wellnessOverview: 'நல்வாழ்வுக் கண்ணோட்டம்',
    metricComparison: 'அளவீட்டு ஒப்பீடு',
    healthRadar: 'சுகாதார ரேடார்',
    comprehensiveView: 'விரிவான பார்வை',
    
    // Messages
    welcomeMessage: 'MindGuard-க்கு வரவேற்கிறோம்',
    assessmentComplete: 'மதிப்பீடு முடிந்தது',
    dataSaved: 'தரவு வெற்றிகரமாகச் சேமிக்கப்பட்டது',
    errorOccurred: 'ஒரு பிழை ஏற்பட்டது',
    noDataAvailable: 'தரவு இல்லை',
    
    // Validation
    required: 'இந்த புலம் தேவை',
    invalidEmail: 'சரியான மின்னஞ்சல் முகவரியை உள்ளிடவும்',
    invalidAge: '13 மற்றும் 120 வயதுக்கு இடையில் சரியான வயதை உள்ளிடவும்',
    passwordTooShort: 'கடவுச்சொல் குறைந்தபட்சம் 8 எழுத்துகள் நீளமாக இருக்க வேண்டும்',
    
    // Actions
    startAssessment: 'மதிப்பீட்டைத் தொடங்கு',
    continueAssessment: 'மதிப்பீட்டைத் தொடரவும்',
    viewResults: 'முடிவுகளைக் காண்க',
    downloadReport: 'அறிக்கையைப் பதிவிறக்கு',
    shareResults: 'முடிவுகளைப் பகிர்',
  },
  
  es: {
    // Common
    title: 'MindGuard',
    subtitle: 'Predicción de Riesgos de Salud con IA',
    loading: 'Cargando...',
    error: 'Error',
    success: 'Éxito',
    cancel: 'Cancelar',
    save: 'Guardar',
    delete: 'Eliminar',
    edit: 'Editar',
    add: 'Agregar',
    search: 'Buscar',
    filter: 'Filtrar',
    sort: 'Ordenar',
    next: 'Siguiente',
    previous: 'Anterior',
    submit: 'Enviar',
    back: 'Atrás',
    close: 'Cerrar',
    
    // Navigation
    dashboard: 'Panel',
    assessment: 'Evaluación',
    history: 'Historial',
    research: 'Investigación',
    profile: 'Perfil',
    settings: 'Configuración',
    logout: 'Cerrar Sesión',
    login: 'Iniciar Sesión',
    register: 'Registrarse',
    
    // Assessment
    healthAssessment: 'Evaluación de Salud',
    basicInformation: 'Información Básica',
    physicalHealth: 'Salud Física',
    mentalWellbeing: 'Bienestar Mental',
    lifestyleFactors: 'Factores de Estilo de Vida',
    reviewSubmit: 'Revisar y Enviar',
    fullName: 'Nombre Completo',
    age: 'Edad',
    gender: 'Género',
    occupation: 'Ocupación',
    email: 'Correo Electrónico',
    sleepHours: 'Horas de Sueño',
    exerciseFrequency: 'Frecuencia de Ejercicio',
    dietQuality: 'Calidad de la Dieta',
    stressLevel: 'Nivel de Estrés',
    moodScore: 'Puntuación del Estado de Ánimo',
    mentalHealthHistory: 'Historial de Salud Mental',
    socialConnections: 'Conexiones Sociales',
    workLifeBalance: 'Equilibrio Trabajo-Vida',
    financialStress: 'Estrés Financiero',
    additionalNotes: 'Notas Adicionales',
    
    // PHQ-9 Screening
    mentalHealthScreening: 'Evaluación de Salud Mental',
    overLastTwoWeeks: 'En las últimas 2 semanas, ¿con qué frecuencia has estado molesto por alguno de los siguientes problemas?',
    notAtAll: 'Para nada',
    severalDays: 'Varios días',
    moreThanHalf: 'Más de la mitad de los días',
    nearlyEveryDay: 'Casi todos los días',
    
    // Dashboard
    overallHealthScore: 'Puntuación General de Salud',
    riskLevel: 'Nivel de Riesgo',
    recommendations: 'Recomendaciones',
    brainHealActivities: 'Actividades de Sanación Cerebral',
    weeklyPlan: 'Plan Semanal',
    progressTracking: 'Seguimiento del Progreso',
    
    // Risk Levels
    low: 'Bajo',
    normal: 'Normal',
    high: 'Alto',
    excellent: 'Excelente',
    good: 'Bueno',
    fair: 'Regular',
    needsAttention: 'Necesita Atención',
    
    // Health Metrics
    sleep: 'Sueño',
    exercise: 'Ejercicio',
    stress: 'Estrés',
    mood: 'Estado de Ánimo',
    energy: 'Energía',
    social: 'Social',
    diet: 'Dieta',
    
    // Time Ranges
    days7: '7 Días',
    days30: '30 Días',
    days90: '90 Días',
    year1: '1 Año',
    
    // Chart Types
    trendAnalysis: 'Análisis de Tendencias',
    wellnessOverview: 'Vista General del Bienestar',
    metricComparison: 'Comparación de Métricas',
    healthRadar: 'Radar de Salud',
    comprehensiveView: 'Vista Integral',
    
    // Messages
    welcomeMessage: 'Bienvenido a MindGuard',
    assessmentComplete: 'Evaluación Completada',
    dataSaved: 'Datos guardados exitosamente',
    errorOccurred: 'Ocurrió un error',
    noDataAvailable: 'No hay datos disponibles',
    
    // Validation
    required: 'Este campo es obligatorio',
    invalidEmail: 'Por favor ingrese una dirección de correo válida',
    invalidAge: 'Por favor ingrese una edad válida entre 13 y 120',
    passwordTooShort: 'La contraseña debe tener al menos 8 caracteres',
    
    // Actions
    startAssessment: 'Comenzar Evaluación',
    continueAssessment: 'Continuar Evaluación',
    viewResults: 'Ver Resultados',
    downloadReport: 'Descargar Reporte',
    shareResults: 'Compartir Resultados',
  },
  
  fr: {
    // Common
    title: 'MindGuard',
    subtitle: 'Prédiction des Risques de Santé par IA',
    loading: 'Chargement...',
    error: 'Erreur',
    success: 'Succès',
    cancel: 'Annuler',
    save: 'Sauvegarder',
    delete: 'Supprimer',
    edit: 'Modifier',
    add: 'Ajouter',
    search: 'Rechercher',
    filter: 'Filtrer',
    sort: 'Trier',
    next: 'Suivant',
    previous: 'Précédent',
    submit: 'Soumettre',
    back: 'Retour',
    close: 'Fermer',
    
    // Navigation
    dashboard: 'Tableau de Bord',
    assessment: 'Évaluation',
    history: 'Historique',
    research: 'Recherche',
    profile: 'Profil',
    settings: 'Paramètres',
    logout: 'Déconnexion',
    login: 'Connexion',
    register: 'S\'inscrire',
    
    // Assessment
    healthAssessment: 'Évaluation de Santé',
    basicInformation: 'Informations de Base',
    physicalHealth: 'Santé Physique',
    mentalWellbeing: 'Bien-être Mental',
    lifestyleFactors: 'Facteurs de Mode de Vie',
    reviewSubmit: 'Réviser et Soumettre',
    fullName: 'Nom Complet',
    age: 'Âge',
    gender: 'Genre',
    occupation: 'Profession',
    email: 'Adresse E-mail',
    sleepHours: 'Heures de Sommeil',
    exerciseFrequency: 'Fréquence d\'Exercice',
    dietQuality: 'Qualité de l\'Alimentation',
    stressLevel: 'Niveau de Stress',
    moodScore: 'Score d\'Humeur',
    mentalHealthHistory: 'Antécédents de Santé Mentale',
    socialConnections: 'Connexions Sociales',
    workLifeBalance: 'Équilibre Travail-Vie',
    financialStress: 'Stress Financier',
    additionalNotes: 'Notes Supplémentaires',
    
    // PHQ-9 Screening
    mentalHealthScreening: 'Dépistage de Santé Mentale',
    overLastTwoWeeks: 'Au cours des 2 dernières semaines, à quelle fréquence avez-vous été dérangé par l\'un des problèmes suivants?',
    notAtAll: 'Pas du tout',
    severalDays: 'Plusieurs jours',
    moreThanHalf: 'Plus de la moitié des jours',
    nearlyEveryDay: 'Presque tous les jours',
    
    // Dashboard
    overallHealthScore: 'Score de Santé Global',
    riskLevel: 'Niveau de Risque',
    recommendations: 'Recommandations',
    brainHealActivities: 'Activités de Guérison Cérébrale',
    weeklyPlan: 'Plan Hebdomadaire',
    progressTracking: 'Suivi des Progrès',
    
    // Risk Levels
    low: 'Faible',
    normal: 'Normal',
    high: 'Élevé',
    excellent: 'Excellent',
    good: 'Bon',
    fair: 'Moyen',
    needsAttention: 'Nécessite une Attention',
    
    // Health Metrics
    sleep: 'Sommeil',
    exercise: 'Exercice',
    stress: 'Stress',
    mood: 'Humeur',
    energy: 'Énergie',
    social: 'Social',
    diet: 'Alimentation',
    
    // Time Ranges
    days7: '7 Jours',
    days30: '30 Jours',
    days90: '90 Jours',
    year1: '1 An',
    
    // Chart Types
    trendAnalysis: 'Analyse des Tendances',
    wellnessOverview: 'Aperçu du Bien-être',
    metricComparison: 'Comparaison des Métriques',
    healthRadar: 'Radar de Santé',
    comprehensiveView: 'Vue Complète',
    
    // Messages
    welcomeMessage: 'Bienvenue sur MindGuard',
    assessmentComplete: 'Évaluation Terminée',
    dataSaved: 'Données sauvegardées avec succès',
    errorOccurred: 'Une erreur s\'est produite',
    noDataAvailable: 'Aucune donnée disponible',
    
    // Validation
    required: 'Ce champ est obligatoire',
    invalidEmail: 'Veuillez saisir une adresse e-mail valide',
    invalidAge: 'Veuillez saisir un âge valide entre 13 et 120',
    passwordTooShort: 'Le mot de passe doit contenir au moins 8 caractères',
    
    // Actions
    startAssessment: 'Commencer l\'Évaluation',
    continueAssessment: 'Continuer l\'Évaluation',
    viewResults: 'Voir les Résultats',
    downloadReport: 'Télécharger le Rapport',
    shareResults: 'Partager les Résultats',
  },

  zh: {
    // Common
    title: 'MindGuard',
    subtitle: '人工智能健康风险预测',
    loading: '加载中...',
    error: '错误',
    success: '成功',
    cancel: '取消',
    save: '保存',
    delete: '删除',
    edit: '编辑',
    add: '添加',
    search: '搜索',
    filter: '筛选',
    sort: '排序',
    next: '下一步',
    previous: '上一步',
    submit: '提交',
    back: '返回',
    close: '关闭',
    
    // Navigation
    dashboard: '仪表板',
    assessment: '评估',
    history: '历史记录',
    research: '研究',
    profile: '个人资料',
    settings: '设置',
    logout: '登出',
    login: '登录',
    register: '注册',
    
    // Assessment
    healthAssessment: '健康评估',
    basicInformation: '基本信息',
    physicalHealth: '身体健康',
    mentalWellbeing: '心理健康',
    lifestyleFactors: '生活方式因素',
    reviewSubmit: '审查并提交',
    fullName: '全名',
    age: '年龄',
    gender: '性别',
    occupation: '职业',
    email: '电子邮件地址',
    sleepHours: '睡眠小时',
    exerciseFrequency: '运动频率',
    dietQuality: '饮食质量',
    stressLevel: '压力水平',
    moodScore: '情绪得分',
    mentalHealthHistory: '心理健康史',
    socialConnections: '社交联系',
    workLifeBalance: '工作与生活平衡',
    financialStress: '财务压力',
    additionalNotes: '附加说明',
    
    // PHQ-9 Screening
    mentalHealthScreening: '心理健康筛查',
    overLastTwoWeeks: '在过去的2周里，你有多少次被以下任何问题困扰？',
    notAtAll: '完全没有',
    severalDays: '几天',
    moreThanHalf: '超过一半的日子',
    nearlyEveryDay: '几乎每天',
    
    // Dashboard
    overallHealthScore: '总体健康得分',
    riskLevel: '风险水平',
    recommendations: '建议',
    brainHealActivities: '大脑康复活动',
    weeklyPlan: '每周计划',
    progressTracking: '进度跟踪',
    
    // Risk Levels
    low: '低',
    normal: '正常',
    high: '高',
    excellent: '优秀',
    good: '良好',
    fair: '一般',
    needsAttention: '需要注意',
    
    // Health Metrics
    sleep: '睡眠',
    exercise: '运动',
    stress: '压力',
    mood: '情绪',
    energy: '精力',
    social: '社交',
    diet: '饮食',
    
    // Time Ranges
    days7: '7天',
    days30: '30天',
    days90: '90天',
    year1: '1年',
    
    // Chart Types
    trendAnalysis: '趋势分析',
    wellnessOverview: '健康概览',
    metricComparison: '指标比较',
    healthRadar: '健康雷达图',
    comprehensiveView: '综合视图',
    
    // Messages
    welcomeMessage: '欢迎来到 MindGuard',
    assessmentComplete: '评估完成',
    dataSaved: '数据保存成功',
    errorOccurred: '发生错误',
    noDataAvailable: '无可用数据',
    
    // Validation
    required: '此字段为必填项',
    invalidEmail: '请输入有效的电子邮件地址',
    invalidAge: '请输入13至120之间的有效年龄',
    passwordTooShort: '密码长度必须至少为8个字符',
    
    // Actions
    startAssessment: '开始评估',
    continueAssessment: '继续评估',
    viewResults: '查看结果',
    downloadReport: '下载报告',
    shareResults: '分享结果',
  },
};

export function getTranslation(language: Language, key: keyof TranslationKeys): string {
  // Use a type assertion to handle the case where a key might not exist on a specific language object, though the structure implies it will.
  const langTranslations = translations[language] as TranslationKeys | undefined;
  return langTranslations?.[key] || translations.en[key] || key;
}

export function getLanguageFromLocale(locale: string): Language {
  const langMap: Record<string, Language> = {
    'en': 'en',
    'en-US': 'en',
    'en-GB': 'en',
    'si': 'si',
    'si-LK': 'si',
    'ta': 'ta',
    'ta-LK': 'ta',
    'ta-IN': 'ta',
    'es': 'es',
    'es-ES': 'es',
    'es-MX': 'es',
    'fr': 'fr',
    'fr-FR': 'fr',
    'fr-CA': 'fr',
    'zh': 'zh',
    'zh-CN': 'zh',
    'zh-TW': 'zh',
  };
  
  return langMap[locale] || 'en';
}

export function getSupportedLanguages(): Array<{ code: Language; name: string; flag: string }> {
  return [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'si', name: 'සිංහල', flag: '🇱🇰' },
    { code: 'ta', name: 'தமிழ்', flag: '🇱🇰' },
    { code: 'es', name: 'Español', flag: '🇪🇸' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'zh', name: '中文', flag: '🇨🇳' },
  ];
}

export function getLanguageName(language: Language): string {
  const languages = getSupportedLanguages();
  return languages.find(lang => lang.code === language)?.name || 'English';
}

export function getLanguageFlag(language: Language): string {
  const languages = getSupportedLanguages();
  return languages.find(lang => lang.code === language)?.flag || '🇺🇸';
}