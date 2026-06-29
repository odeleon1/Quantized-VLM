interface Props {
  password: string;
}

interface Score {
  score: number;
  label: string;
  color: string;
}

function scorePassword(p: string): Score {
  let s = 0;
  if (p.length >= 8)            s++;
  if (p.length >= 12)           s++;
  if (/[A-Z]/.test(p))         s++;
  if (/[a-z]/.test(p))         s++;
  if (/[0-9]/.test(p))         s++;
  if (/[^A-Za-z0-9]/.test(p))  s++;

  const labels = ["Very Weak", "Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"];
  const colors = ["#c0392b", "#c0392b", "#e67e22", "#f39c12", "#27ae60", "#1a8a4a", "#0d6b35"];
  return { score: s, label: labels[s], color: colors[s] };
}

export function PasswordStrengthBar({ password }: Props) {
  if (!password) return null;

  const { score, label, color } = scorePassword(password);
  const pct = Math.round((score / 6) * 100);

  return (
    <div className="strength-bar-wrap">
      <div className="strength-bar-track">
        <div
          className="strength-bar-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="strength-label" style={{ color }}>{label}</span>
    </div>
  );
}
