export default function Toast({ toast }) {
  return (
    <div className={`toast${toast ? ` show ${toast.type}` : ''}`}>
      {toast?.msg}
    </div>
  );
}
