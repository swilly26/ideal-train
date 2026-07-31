import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const DATA_DIR = "/home/team/shared/data";
const DB_PATH = join(DATA_DIR, "apextrade_users.json");

export interface User {
  id: number;
  email: string;
  password_hash: string;
  role: "user" | "admin";
  created_at: string;
  alpacaKey: string | null;       // AES-256-GCM encrypted
  alpacaSecret: string | null;    // AES-256-GCM encrypted
  alpacaConnected: boolean;
  alpacaPaperTrading: boolean;
}

interface Store {
  users: User[];
  nextId: number;
}

function readStore(): Store {
  if (!existsSync(DB_PATH)) {
    return { users: [], nextId: 1 };
  }
  const raw = readFileSync(DB_PATH, "utf-8");
  const store = JSON.parse(raw) as Store;
  // Migrate existing users that lack new fields
  for (const u of store.users) {
    if (!("role" in u)) (u as any).role = "user";
    if (!("alpacaKey" in u)) (u as any).alpacaKey = null;
    if (!("alpacaSecret" in u)) (u as any).alpacaSecret = null;
    if (!("alpacaConnected" in u)) (u as any).alpacaConnected = false;
    if (!("alpacaPaperTrading" in u)) (u as any).alpacaPaperTrading = true;
  }
  return store;
}

function writeStore(store: Store): void {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }
  writeFileSync(DB_PATH, JSON.stringify(store, null, 2), "utf-8");
}

export function findUserByEmail(email: string): User | undefined {
  const store = readStore();
  return store.users.find((u) => u.email === email);
}

export function findUserById(id: number): User | undefined {
  const store = readStore();
  return store.users.find((u) => u.id === id);
}

export function getAllUsers(): User[] {
  const store = readStore();
  return store.users;
}

export function createUser(
  email: string,
  passwordHash: string,
  role: "user" | "admin" = "user",
): User {
  const store = readStore();
  const now = new Date().toISOString();
  const user: User = {
    id: store.nextId++,
    email,
    password_hash: passwordHash,
    role,
    created_at: now,
    alpacaKey: null,
    alpacaSecret: null,
    alpacaConnected: false,
    alpacaPaperTrading: true,
  };
  store.users.push(user);
  writeStore(store);
  return user;
}

export function updateUser(
  id: number,
  updates: Partial<Pick<User, "role" | "alpacaKey" | "alpacaSecret" | "alpacaConnected" | "alpacaPaperTrading">>,
): User | undefined {
  const store = readStore();
  const user = store.users.find((u) => u.id === id);
  if (!user) return undefined;
  Object.assign(user, updates);
  writeStore(store);
  return user;
}

export function updateUserRole(id: number, role: "user" | "admin"): User | undefined {
  return updateUser(id, { role });
}
