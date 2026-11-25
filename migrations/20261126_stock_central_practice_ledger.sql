-- Dedicated stock/trading coordination ledger.
-- Apply only to the separate "stock price" Supabase project.
-- User UUIDs are opaque IDs from Torn-Helper auth; this database does not own auth.
begin;
create schema if not exists stock_central;
create table if not exists stock_central.practice_accounts (
  user_id uuid primary key,
  cash numeric not null default 1000000 check (cash >= 0),
  realized_pl numeric not null default 0,
  started_at timestamptz not null default now(),
  migrated_at timestamptz not null default now()
);
create table if not exists stock_central.practice_positions (
  user_id uuid not null references stock_central.practice_accounts(user_id),
  stock_id smallint not null check (stock_id between 1 and 35),
  shares bigint not null check (shares > 0),
  avg_cost numeric not null check (avg_cost > 0),
  primary key(user_id,stock_id)
);
create table if not exists stock_central.practice_orders (
  user_id uuid not null references stock_central.practice_accounts(user_id),
  request_id uuid not null,
  stock_id smallint not null check (stock_id between 1 and 35),
  side text not null check (side in ('buy','sell')),
  shares bigint not null check (shares > 0),
  price numeric not null check (price > 0),
  fee numeric not null default 0 check (fee >= 0),
  realized_pl numeric not null default 0,
  quote_at timestamptz not null,
  created_at timestamptz not null default now(),
  primary key(user_id,request_id)
);
create table if not exists stock_central.migration_audit (
  migration_key text primary key,
  source_project text not null,
  source_table text not null,
  source_row_count bigint not null,
  copied_row_count bigint not null,
  verified_at timestamptz not null default now(),
  details jsonb not null default '{}'::jsonb,
  check (source_row_count=copied_row_count)
);
alter table stock_central.practice_accounts enable row level security;
alter table stock_central.practice_positions enable row level security;
alter table stock_central.practice_orders enable row level security;
alter table stock_central.migration_audit enable row level security;
revoke all on schema stock_central from public, anon, authenticated;
revoke all on all tables in schema stock_central from public, anon, authenticated;
grant usage on schema stock_central to service_role;
create or replace function stock_central.practice_snapshot(p_user_id uuid)
returns jsonb language sql security definer
set search_path = pg_catalog, stock_central
as $$
 select jsonb_build_object(
  'cash',a.cash,'realized_pl',a.realized_pl,'started_at',a.started_at,
  'positions',coalesce((select jsonb_agg(jsonb_build_object('stock_id',p.stock_id,'shares',p.shares,'avg_cost',p.avg_cost) order by p.stock_id) from stock_central.practice_positions p where p.user_id=a.user_id),'[]'::jsonb),
  'orders',coalesce((select jsonb_agg(to_jsonb(o) order by o.created_at desc) from stock_central.practice_orders o where o.user_id=a.user_id),'[]'::jsonb)
 ) from stock_central.practice_accounts a where a.user_id=p_user_id
$$;
create or replace function stock_central.trade_practice_order(
 p_user_id uuid,p_stock_id integer,p_side text,p_shares bigint,p_request_id uuid,p_price numeric,p_quote_at timestamptz
) returns jsonb language plpgsql security definer
set search_path = pg_catalog, stock_central
as $$
declare balance numeric;held bigint:=0;cost numeric:=0;gross numeric;fee numeric:=0;profit numeric:=0;existing stock_central.practice_orders%rowtype;
begin
 if p_user_id is null then raise exception 'Sign in to use practice trading.';end if;
 if p_stock_id is null or p_stock_id not between 1 and 35 or p_side not in ('buy','sell') or p_shares is null or p_shares<=0 or p_shares>1000000000 or p_request_id is null then raise exception 'Choose a stock and a positive whole number of shares.';end if;
 insert into stock_central.practice_accounts(user_id) values(p_user_id) on conflict do nothing;
 select cash into balance from stock_central.practice_accounts where user_id=p_user_id for update;
 select * into existing from stock_central.practice_orders where user_id=p_user_id and request_id=p_request_id;
 if found then
  if existing.stock_id<>p_stock_id or existing.side<>p_side or existing.shares<>p_shares then raise exception 'Order reference already used.';end if;
  return stock_central.practice_snapshot(p_user_id);
 end if;
 if p_price is null or p_price<=0 or p_quote_at is null or p_quote_at<now()-interval '2 minutes' or p_quote_at>now() or p_price>=1e12 then raise exception 'Wait for a fresh Torn price before trading.';end if;
 select shares,avg_cost into held,cost from stock_central.practice_positions where user_id=p_user_id and stock_id=p_stock_id;
 held:=coalesce(held,0);cost:=coalesce(cost,0);gross:=round(p_price*p_shares,8);
 if p_side='buy' then
  if gross>balance then raise exception 'Not enough practice cash.';end if;
  update stock_central.practice_accounts set cash=cash-gross where user_id=p_user_id;
  insert into stock_central.practice_positions(user_id,stock_id,shares,avg_cost) values(p_user_id,p_stock_id,p_shares,p_price)
  on conflict(user_id,stock_id) do update set avg_cost=(stock_central.practice_positions.avg_cost*stock_central.practice_positions.shares+gross)/(stock_central.practice_positions.shares+p_shares),shares=stock_central.practice_positions.shares+p_shares;
 else
  if p_shares>held then raise exception 'You do not own enough practice shares.';end if;
  fee:=round(gross*.001,8);profit:=gross-fee-cost*p_shares;
  update stock_central.practice_accounts set cash=cash+gross-fee,realized_pl=realized_pl+profit where user_id=p_user_id;
  if held=p_shares then delete from stock_central.practice_positions where user_id=p_user_id and stock_id=p_stock_id;
  else update stock_central.practice_positions set shares=shares-p_shares where user_id=p_user_id and stock_id=p_stock_id;end if;
 end if;
 insert into stock_central.practice_orders(user_id,request_id,stock_id,side,shares,price,fee,realized_pl,quote_at)
 values(p_user_id,p_request_id,p_stock_id,p_side,p_shares,p_price,fee,profit,p_quote_at);
 return stock_central.practice_snapshot(p_user_id);
end
$$;
create or replace function stock_central.reset_practice_account(p_user_id uuid)
returns jsonb language plpgsql security definer
set search_path = pg_catalog, stock_central
as $$
begin
 if p_user_id is null then raise exception 'Sign in to restart practice trading.';end if;
 insert into stock_central.practice_accounts(user_id) values(p_user_id) on conflict do nothing;
 perform 1 from stock_central.practice_accounts where user_id=p_user_id for update;
 delete from stock_central.practice_orders where user_id=p_user_id;
 delete from stock_central.practice_positions where user_id=p_user_id;
 update stock_central.practice_accounts set cash=1000000,realized_pl=0,started_at=now() where user_id=p_user_id;
 return stock_central.practice_snapshot(p_user_id);
end
$$;
revoke all on function stock_central.practice_snapshot(uuid) from public,anon,authenticated;
revoke all on function stock_central.trade_practice_order(uuid,integer,text,bigint,uuid,numeric,timestamptz) from public,anon,authenticated;
revoke all on function stock_central.reset_practice_account(uuid) from public,anon,authenticated;
grant execute on function stock_central.practice_snapshot(uuid) to service_role;
grant execute on function stock_central.trade_practice_order(uuid,integer,text,bigint,uuid,numeric,timestamptz) to service_role;
grant execute on function stock_central.reset_practice_account(uuid) to service_role;
commit;
