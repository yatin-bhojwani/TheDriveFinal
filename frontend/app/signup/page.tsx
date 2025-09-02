    'use client';

    import { useState } from 'react';
    import { useRouter } from 'next/navigation';
    import Link from 'next/link';
    import { api } from '@/lib/api';

    export default function SignupPage() {
        const router = useRouter();
        const [error, setError] = useState<string | null>(null);
        const [success, setSuccess] = useState<string | null>(null);
        const [isLoading, setIsLoading] = useState(false);

        const handleSignup = async (event: React.FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            setIsLoading(true);
            setError(null);
            setSuccess(null);
            
            const formData = new FormData(event.currentTarget);
            const email = formData.get('email') as string;
            const password = formData.get('password') as string;
            const confirmPassword = formData.get('confirmPassword') as string;

            if (password !== confirmPassword) {
                setError("Passwords do not match.");
                setIsLoading(false);
                return;
            }

            try {
                await api.signup(email, password);
                setSuccess("Account created successfully! Redirecting to login...");
                setTimeout(() => router.push('/login'), 2000);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setIsLoading(false);
            }
        };

        return (
            <div className="flex items-center justify-center h-screen bg-gray-100 dark:bg-gray-900">
                <div className="w-full max-w-md p-8 space-y-8 bg-white rounded-lg shadow-md dark:bg-gray-800">
                    <h1 className="text-2xl font-bold text-center text-gray-900 dark:text-white">Create an Account</h1>
                    <form className="space-y-6" onSubmit={handleSignup}>
                        <div>
                            <label htmlFor="email" className="text-sm font-medium text-gray-700 dark:text-gray-300">Email address</label>
                            <input id="email" name="email" type="email" autoComplete="email" required className="w-full px-3 py-2 mt-1 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        </div>
                        <div>
                            <label htmlFor="password"className="text-sm font-medium text-gray-700 dark:text-gray-300">Password</label>
                            <input id="password" name="password" type="password" required className="w-full px-3 py-2 mt-1 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        </div>
                        <div>
                            <label htmlFor="confirmPassword"className="text-sm font-medium text-gray-700 dark:text-gray-300">Confirm Password</label>
                            <input id="confirmPassword" name="confirmPassword" type="password" required className="w-full px-3 py-2 mt-1 border rounded-md dark:bg-gray-700 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" />
                        </div>
                        {error && <p className="text-sm text-red-500">{error}</p>}
                        {success && <p className="text-sm text-green-500">{success}</p>}
                        <button type="submit" disabled={isLoading} className="w-full px-4 py-2 font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-blue-400">
                            {isLoading ? 'Creating account...' : 'Sign up'}
                        </button>
                    </form>
                    <p className="text-sm text-center text-gray-600 dark:text-gray-400">
                        Already have an account?{' '}
                        <Link href="/login" className="font-medium text-blue-600 hover:underline dark:text-blue-500">
                            Sign in
                        </Link>
                    </p>
                </div>
            </div>
        );
    }